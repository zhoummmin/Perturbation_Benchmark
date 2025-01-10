from copy import deepcopy
import argparse
from time import time
import sys, os
import pickle

import scanpy as sc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch_geometric.data import DataLoader

from .model import *
from .inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis, compute_synergy_loss
from .utils import loss_fct, uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params
from .scfm_utils import *

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

class GEARS:
    def __init__(self, pert_data, 
                 local_rank,
                 is_master: bool,
                 world_size,
                 train_bs = 48,
                 test_bs = 48,
                 device = 'cuda',
                 weight_bias_track = True, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS',):
        
        self.weight_bias_track = weight_bias_track
        self.is_master = is_master
        self.local_rank = local_rank
        self.world_size = world_size
        self.proj_name = proj_name
        self.exp_name = exp_name
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  # zhoumin
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = f'cuda:{local_rank}'
        torch.cuda.set_device(self.device)   # zhoumin
        self.config = None
        
        self.dataloader = {}
        for dl_key, dl in pert_data.dataloader.items():
            dataset = dl.dataset
            if 'train' in dl_key:
                sampler = DistributedSampler(dataset)
                self.dataloader[dl_key] = DataLoader(dataset, batch_size=train_bs, sampler=sampler)
            elif 'val' in dl_key:
                sampler = SequentialDistributedSampler(dataset, batch_size=test_bs, world_size=self.world_size)
                self.dataloader[dl_key] = DataLoader(dataset, batch_size=test_bs, sampler=sampler)
            else:
                self.dataloader[dl_key] = dl
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = list(pert_data.gene_names)
        self.num_genes = len(self.gene_list)
        try:
            self.real_gene_list = list(pert_data.real_gene_names)
        except AttributeError:
            self.real_gene_list = list(pert_data.gene_names)
        self.real_num_genes = len(self.real_gene_list)
        if self.num_genes != self.real_num_genes:
            print_sys("The gene list is augmented, larger than real gene list!")
        self.ctrl_expression = torch.tensor(np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'], axis = 0)).reshape(-1,).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i,j in 
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']

        #For loss analyzation: 
        self.train_losses = []
        self.train_overall_mses = []
        self.val_overall_mses = []
        self.train_de_mses = []
        self.val_de_mses = []
        
    def tunable_parameters(self):
        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False, 
                         cell_fitness_pred = False,
                         go_path = None,
                         model_type=None,
                         bin_set=None,
                         load_path=None,
                         finetune_method=None,
                         accumulation_steps=1,
                         mode='v1',
                         input_genes_ens_ids = None,
                         scfm_genes_ens_ids = None,
                         scfm_hyper_params_path = None,
                         scfm_ckpt_path = 'None',
                         scfm_class = PerformerLM_GO, 
                         model_class = GEARS_Model_Acc,
                         key_enc = None,
                         **kwargs
                        ):
        
        self.config = {'proj_name': self.proj_name,
                       'exp_name': self.exp_name,
                        'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'no_perturb': no_perturb,
                       'cell_fitness_pred': cell_fitness_pred,
                        'model_type': model_type,
                        'bin_set': bin_set,
                        'load_path': load_path,
                        'finetune_method': finetune_method,
                        'accumulation_steps': accumulation_steps,
                        'mode':mode,
                        'input_genes_ens_ids': input_genes_ens_ids,
                        'scfm_genes_ens_ids': scfm_genes_ens_ids,
                        "scfm_hyper_params_path": scfm_hyper_params_path,
                        "scfm_ckpt": scfm_ckpt_path,
                        "scfm_class": scfm_class,
                        "key_enc": key_enc
                      }
        print('Use accumulation steps:',accumulation_steps)
        print('Use mode:',mode)
        self.config.update(kwargs)
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type = 'co-express', adata = self.adata, threshold = coexpress_threshold, k = num_similar_genes_co_express_graph, gene_list = self.gene_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed = self.seed, train_gene_set_size = self.train_gene_set_size, set2conditions = self.set2conditions)
            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type = 'go', adata = self.adata, threshold = coexpress_threshold, k = num_similar_genes_go_graph, gene_list = self.gene_list, data_path = self.data_path, data_name = self.dataset_name, split = self.split, seed = self.seed, train_gene_set_size = self.train_gene_set_size, set2conditions = self.set2conditions)
            print_sys(f"The go_edge_list shape: {edge_list.shape}")
            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = model_class(self.config).to(self.device)
        
    def load_pretrained(self, path):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.model.module.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, pert_list):
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.
        
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " not in the perturbation graph. Please select from PertNet.gene_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        from torch_geometric.data import DataLoader
        for pert in pert_list:
            try:
                #If prediction is already saved, then skip inference
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
                if self.config['uncertainty']:
                    results_logvar_sum['_'.join(pert)] = self.saved_logvar_sum['_'.join(pert)]
                continue
            except:
                pass
            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata, self.pert_list, self.device)
            loader = DataLoader(cg, 5, shuffle = False)
            # batch = next(iter(loader))
            # batch.to(self.device)
            
            predall=[]
            for step, batch in enumerate(loader):
                batch.to(self.device)
                with torch.no_grad():
                    if self.config['uncertainty']:
                        p, unc = self.best_model(batch)
                        results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                        results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                    else:
                        p = self.best_model(batch)
                    predall.append(p.detach().cpu().numpy())
            preadall = np.concatenate(predall,axis=0)
            # results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
            results_pred['_'.join(pert)] = np.mean(preadall, axis = 0)
            
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        ## given a gene pair, return (1) transcriptome of A,B,A+B and (2) GI scores. 
        ## if uncertainty mode is on, also return uncertainty score.
        
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[combo[0]]
            pred[combo[1]] = self.saved_pred[combo[1]]
            pred['_'.join(combo)] = self.saved_pred['_'.join(combo)]
        except:
            if self.config['uncertainty']:
                pred = self.predict([[combo[0]], [combo[1]], combo])[0]
            else:
                pred = self.predict([[combo[0]], [combo[1]], combo])

        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 

        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
            
        pred = {p:pred[p][GI_genes_idx] for p in pred}
        return get_GI_params(pred, combo)
    
    def plot_perturbation(self, query, save_file = None):
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
        
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        
        de_idx = [gene2idx[gene_raw2id[i]] for i in adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        pred = self.predict([query.split('+')])['_'.join(query.split('+'))][de_idx]
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[de_idx].values
        
        pred = pred - ctrl_means
        truth = truth - ctrl_means
        
        plt.figure(figsize=[16.5,4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False,
                    medianprops = dict(linewidth=0))    

        for i in range(pred.shape[0]):
            _ = plt.scatter(i+1, pred[i], color='red')

        plt.axhline(0, linestyle="dashed", color = 'green')

        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation = 90)

        plt.ylabel("Change in Gene Expression over Control",labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        plt.show()
        
    def load_pretrained(self, path, gene_emb):
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        config['gene_emb'] = gene_emb
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def train(self, 
              epochs = 20, 
              result_dir='./results',
              lr = 1e-3,
              weight_decay = 5e-4, 
              valid_every = 1
             ):
        GRADIENT_ACCUMULATION = self.config['accumulation_steps']
        test = (epochs == 1)
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model

        if self.config['finetune_method'] == 'frozen':
            for name, p in self.model.named_parameters():
                if "singlecell_model" in name:
                    p.requires_grad = False
        else:
            for name, p in self.model.named_parameters():
                p.requires_grad = True

        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        dist.barrier()
        if self.is_master:
            print_sys('Start Training...')
        for epoch in range(epochs):
            train_loader.sampler.set_epoch(epoch)
            self.model.train()
            dist.barrier()
            for step, batch in enumerate(train_loader):
                if test and (step+1 > 2 * GRADIENT_ACCUMULATION):
                    break
                batch.to(self.device)
                y = batch.y
                if (step+1) % GRADIENT_ACCUMULATION != 0:
                    with self.model.no_sync():
                        pred = self.model(batch)
                        loss = loss_fct(pred, y, batch.pert,
                                    ctrl = self.ctrl_expression, 
                                    dict_filter = self.dict_filter,
                                    direction_lambda = self.config['direction_lambda'])
                    
                        loss.backward()
                elif (step+1) % GRADIENT_ACCUMULATION == 0:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                ctrl = self.ctrl_expression, 
                                dict_filter = self.dict_filter,
                                direction_lambda = self.config['direction_lambda'])
                
                    loss.backward()
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if (step + 1) % 50 == 0:
                    if self.is_master:
                        if self.wandb:
                            self.wandb.log({'training_loss': loss.item()})
                        log = "Epoch {} Step {} Train Loss: {:.4f}" 
                        print_sys(log.format(epoch + 1, step + 1, loss.item()))
            dist.barrier()
            scheduler.step()
            if (epoch + 1) % valid_every == 0:
                # Evaluate model performance on train and val set
                with torch.no_grad():
                    val_res = evaluate(val_loader, self.model, self.config['uncertainty'], self.device)
                val_metrics, _ = compute_metrics(val_res)
                del val_res
                val_metrics_reduced = {}
                for m in ['mse', 'mse_de', 'pearson', 'pearson_de']:
                    val_metrics_reduced[m] = get_reduced(val_metrics[m], self.local_rank, 0, self.world_size)
                val_metrics = val_metrics_reduced
                # Print epoch performance
                log = "Epoch {}: Validation Overall MSE: {:.4f}. Validation Top 20 DE MSE: {:.4f}."

                if self.is_master:
                    print_sys(log.format(epoch + 1, val_metrics['mse'], val_metrics['mse_de']))
                    if self.wandb:
                        metrics = ['mse', 'pearson']
                        for m in metrics:
                            self.wandb.log({'val_'+m: val_metrics[m],'val_de_'+m: val_metrics[m + '_de']})
                    if val_metrics['mse_de'] < min_val:
                        min_val = val_metrics['mse_de']
                        print_sys("Best epoch:{} mse_de:{}!".format(epoch + 1, min_val))
                        self.save_model(result_dir)
            
        dist.barrier()
        if self.is_master:
            print_sys("Done!")
            if 'test_loader' not in self.dataloader:
                print_sys('Done! No test dataloader detected.')
                return
            checkpoint = torch.load(os.path.join(result_dir, 'model.pt'), map_location=self.device)
            self.model.module.load_state_dict(checkpoint)
            self.model.eval()

            if self.config.get('record_pred', False):
                self.model.module.args['start_record'] = True
                
            # Model testing
            test_loader = self.dataloader['test_loader']
            if self.is_master:
                print_sys("Start Testing...")
            with torch.no_grad():
                test_res = evaluate(test_loader, self.model.module, self.config['uncertainty'], self.device)
                '''
                if not test:
                    test_res = evaluate(test_loader, self.model.module, self.config['uncertainty'], self.device)
                else:
                    quick_model = debug_model()
                    test_res = evaluate(test_loader, quick_model, self.config['uncertainty'], self.device)'''
            test_metrics, test_pert_res = compute_metrics(test_res)
            
            '''
            test_metrics_reduced = {}
            for m in test_metrics.keys():
                test_metrics_reduced[m] = get_reduced(test_metrics[m], self.local_rank, 0, self.world_size)
            test_metrics = test_metrics_reduced'''

            log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
            if self.is_master:
                print_sys(log.format(test_metrics['mse_de']))
                if self.wandb:
                    metrics = ['mse', 'pearson']
                    for m in metrics:
                        self.wandb.log({'test_' + m: test_metrics[m],
                                'test_de_'+m: test_metrics[m + '_de']                     
                                })
                    
            out = deeper_analysis(self.adata, test_res)
            out_non_dropout = non_dropout_analysis(self.adata, test_res) 

            if self.split == 'simulation':
                print_sys("Start doing subgroup analysis for simulation split...")
                subgroup = self.subgroup
                subgroup_analysis = {}
                for name in subgroup['test_subgroup'].keys():
                    subgroup_analysis[name] = {}
                    for m in list(list(test_pert_res.values())[0].keys()):
                        subgroup_analysis[name][m] = []

                for name, pert_list in subgroup['test_subgroup'].items():
                    for pert in pert_list:
                        for m, res in test_pert_res[pert].items():
                            subgroup_analysis[name][m].append(res)

                for name, result in subgroup_analysis.items():
                    for m in result.keys():
                        subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                        if self.wandb:
                            self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                        print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

                ## deeper analysis
                subgroup_analysis = {}
                for name in subgroup['test_subgroup'].keys():
                    subgroup_analysis[name] = {}

                for name, pert_list in subgroup['test_subgroup'].items():
                    for pert in pert_list:
                        for m in out[pert].keys():
                            if m in subgroup_analysis[name].keys():
                                subgroup_analysis[name][m].append(out[pert][m])
                            else:
                                subgroup_analysis[name][m] = [out[pert][m]]

                        for m in out_non_dropout[pert].keys():
                            if m in subgroup_analysis[name].keys():
                                subgroup_analysis[name][m].append(out_non_dropout[pert][m])
                            else:
                                subgroup_analysis[name][m] = [out_non_dropout[pert][m]]

                for name, result in subgroup_analysis.items():
                    for m in result.keys():
                        subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                        if self.wandb:
                            self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                        print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
            print_sys('Done!')


            

