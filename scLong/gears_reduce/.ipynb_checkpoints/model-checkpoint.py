import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import numpy as np
import pickle
from torch_geometric.nn import SGConv
import sys
sys.path.append('/home/share/huadjyin/home/zhoumin3/scLong/scLong')
from performer_pytorch_cont.ding_models import PerformerLM_GO, DualEncoderSCFM
from copy import deepcopy
import pickle
import os
import scipy

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)
    


def reindex_tensor_universal(tensor, index_positions, dim, filler = '0', device = 'cpu'):
    """
    Reindex a tensor along a specified dimension using new indices.

    :param tensor: Input tensor
    :param original_indices: List of original indices
    :param new_indices: List of new indices
    :param dim: Dimension along which to reindex
    :filler: Reindex filler, '0' or 'mean'. 
    :return: Reindexed tensor
    """

    # Convert list to tensor
    index_tensor = torch.tensor(index_positions, device=device)
    # Adjust shapes for the gather operation
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = len(index_positions)
    index_shape = [1] * len(tensor_shape)
    index_shape[dim] = len(index_positions)
    index_tensor = index_tensor.view(*index_shape).expand(*tensor_shape)

    if filler == '0':
        padder_shape = deepcopy(tensor_shape)
        padder_shape[dim] = 1
        padder = torch.zeros(padder_shape, dtype = tensor.dtype, device=device)
    elif filler == 'mean':
        padder = tensor.mean(dim = dim, keepdim = True)
    else:
        raise ValueError("filler should be 0 or mean!")
    expanded_tensor = torch.cat((tensor, padder), dim = dim)

    # Reindex tensor
    reindexed = torch.gather(expanded_tensor, dim, index_tensor)

    return reindexed
    
class SCFM_different_genes(torch.nn.Module):
    def __init__(self, input_genes, scfm_genes, 
                 scfm_hyper_params, scfm_ckpt, 
                 scfm_class = PerformerLM_GO,
                 key_enc = None,
                 requires_grad = False, 
                 device = 'cpu',
                 output_dim = 200):
        super(SCFM_different_genes, self).__init__()
        # Create a mapping from new_indices to original_indices
        self.device = device
        input_mapping = {idx: i for i, idx in enumerate(input_genes)}
        self.input_gene_num = len(input_genes)
        # Convert new indices to a list of index positions from the original tensor
        scfm_genes_pad = scfm_genes + ['PAD']
        self.scfm_seq_len = len(scfm_genes_pad)
        scfm_mapping = {idx: i for i, idx in enumerate(scfm_genes_pad)}

        self.scfm_index_positions = [input_mapping.get(idx, self.input_gene_num) for idx in scfm_genes_pad]
        self.input_index_positions = [scfm_mapping.get(idx, self.scfm_seq_len) for idx in input_genes]
        self.key_enc = key_enc
        self.scfm = scfm_class(
            **scfm_hyper_params
        )
        if scfm_ckpt != 'None':
            ckpt = torch.load(scfm_ckpt, map_location = device)
            self.scfm.load_state_dict(ckpt['model_state_dict'])
            for param in self.scfm.parameters():
                param.requires_grad = requires_grad
        else:
            for param in self.scfm.parameters():
                param.requires_grad = True

        if self.scfm_seq_len != self.scfm.max_seq_len:
            raise ValueError("scfm_ckpt not same with scfm_genes list!")
    
        self.output_dim = output_dim
        if self.output_dim != 200:
            try:
                self.feedforward = nn.Linear(self.scfm.base_dim, self.output_dim)
            except AttributeError:
                self.feedforward = nn.Linear(scfm_hyper_params['dim'], self.output_dim)

        
    def forward(self, x):
        x = x.view((-1, self.input_gene_num)) #b, m
        x_scfm = reindex_tensor_universal(x, self.scfm_index_positions, dim = 1, filler = '0', device=self.device) #b, n
        output_encodings = self.scfm(x_scfm, return_encodings = True) #b, n, d or b, n_top, d
        if self.key_enc != None:
            output_encodings = output_encodings[self.key_enc]
        if self.key_enc == "top_encodings":
            if self.output_dim != 200:
                output_encodings = self.feedforward(output_encodings)
            return output_encodings #b, n_top, d
        output_reindex = reindex_tensor_universal(output_encodings, self.input_index_positions, dim = 1, filler = 'mean', device=self.device) #b, m, d
        if self.output_dim != 200:
            output_reindex = self.feedforward(output_reindex)
        return output_reindex
    
class GEARS_Model_Acc(torch.nn.Module):
    """
    GEARS
    """

    def __init__(self, args):
        super(GEARS_Model_Acc, self).__init__()
        self.args = args 

        if args.get('record_pred', False):
            self.pred_dir = f"./result_process/preds/{self.args['exp_name']}"
            if not os.path.exists(self.pred_dir):
                os.mkdir(self.pred_dir)
            self.pred_batch_idx = 0

        self.num_genes = args['num_genes']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.pert_emb_lambda = 0.2

        self.input_genes_list = args['input_genes_ens_ids']
        self.scfm_genes_list = args['scfm_genes_ens_ids']
        try:
            with open(args["scfm_hyper_params_path"], 'rb') as handle:
                self.scfm_hyper_params = pickle.load(handle)
        except ValueError:
            import pickle5
            with open(args["scfm_hyper_params_path"], 'rb') as handle:
                self.scfm_hyper_params = pickle5.load(handle)
        self.scfm_hyper_params['gene2vec_file'] = args['scfm_gene2vec_file']
        self.scfm_ckpt = args["scfm_ckpt"]
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        #self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        '''
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))'''
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')

        if args['model_type'] == 'API':
            # It will be implemented in the future
            def API(x):
                return torch.rand(x.shape[0], x.shape[1]-1,hidden_size).to(x.device)
            self.singlecell_model = API
            self.pretrained = True
        elif args['model_type'] == 'GO_CONT_SCFM':
            self.singlecell_model = SCFM_different_genes(self.input_genes_list, 
                                                         self.scfm_genes_list, 
                                                         self.scfm_hyper_params,
                                                         self.scfm_ckpt,
                                                         scfm_class = args["scfm_class"], 
                                                         key_enc = args["key_enc"], 
                                                         device= args['device'],
                                                         output_dim=hidden_size)
            self.pretrained = True
        else:
            self.pretrained = False
            print('No Single cell model load!')
        

    def forward(self, data):
        if str(type(data)) == "<class 'torch.Tensor'>":
            x = data.reshape((-1, 2))
        else:
            x, batch = data.x, data.batch
        x_pert = x[:, 1].reshape((-1, self.num_genes))
        num_graphs = len(data.batch.unique())

        ## get base gene embeddings
        if self.pretrained:
            exp = x[:, 0].reshape((-1, self.num_genes)) #B, N
            emb = self.singlecell_model(exp) #B, N, D
            emb = emb.reshape((num_graphs * self.num_genes, -1))
        else:
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)        
        
        '''
        pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
        for idx, layer in enumerate(self.layers_emb_pos):
            pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.layers_emb_pos) - 1:
                pos_emb = pos_emb.relu()

        base_emb = base_emb + 0.2 * pos_emb
        base_emb = self.emb_trans_v2(base_emb)'''

        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

        pert = x_pert.reshape(-1,1) #B * N, 1
        #pert_index = torch.where(pert.reshape(num_graphs, int(x.shape[0]/num_graphs)) == 1)

        pert_embeddings = self.pert_emb(torch.LongTensor(list(range(self.num_genes))).to(self.args['device']))      #N, D
            
        ## augment global perturbation embedding with GNN
        for idx, layer in enumerate(self.sim_layers):
            pert_embeddings = layer(pert_embeddings, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_embeddings = pert_embeddings.relu() 

        #After GNN is still N, D
        pert_embeddings = torch.stack(list((pert_embeddings,)) * num_graphs)   #B, N, D
        pert_embeddings = pert_embeddings * x_pert.reshape((num_graphs, self.num_genes, 1)) #B, N, D
        pert_embeddings = pert_embeddings.sum(1) #B, D
        pert_mask = (x_pert.sum(1) != 0).reshape((num_graphs, 1)) #B, 1
        pert_embeddings = self.pert_fuse(pert_embeddings) * pert_mask #B, D
        pert_embeddings = pert_embeddings.reshape((num_graphs, 1, pert_embeddings.shape[1])) #B, 1, D

        base_emb = base_emb + pert_embeddings

        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        
        ## add the perturbation positional embedding
        pert_emb = self.pert_w(pert)
        combined = pert_emb+base_emb
        combined = self.bn_pert_base_trans(combined)
        base_emb = self.pert_base_trans(combined)
        base_emb = self.bn_pert_base(base_emb)
        
        ## apply the first MLP
        base_emb = self.transform(base_emb)        
        out = self.recovery_w(base_emb) #B * N, D
        out = out.reshape(num_graphs, self.num_genes, -1) #B, N, D
        out = out.unsqueeze(-1) * self.indv_w1 #B, N, D, 1
        w = torch.sum(out, axis = 2) ##B, N, 1
        out = w + self.indv_b1 #B, N, 1

        # Cross gene
        cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2)) #B, D
        cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes) #B, N*D

        cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])#B, N, D
        cross_gene_out = torch.cat([out, cross_gene_embed], 2) #B, N, D+1

        cross_gene_out = cross_gene_out * self.indv_w2 #B, N, D+1
        cross_gene_out = torch.sum(cross_gene_out, axis=2) #B, N
        out = cross_gene_out + self.indv_b2        #B, N
        out = out.reshape(num_graphs * self.num_genes, -1) + x[:, 0].reshape(-1,1) #B * N, 1

        ####################
        # Record test predictions #
        ####################

        if self.args.get('start_record', False):
            y_flat = data.y.reshape(-1, 1) #B * N, 1
            pred = torch.cat((pert, out, y_flat), dim = 1) #B * N, 3
            pred = pred.cpu().to_sparse()

            # Convert the sparse tensor to a SciPy sparse matrix (COO format)
            values = pred.values().numpy()
            indices = pred.indices().numpy()
            sparse_matrix = scipy.sparse.coo_matrix((values, indices), shape=pred.shape)

            # Save the sparse matrix in NPZ format
            scipy.sparse.save_npz(f'{self.pred_dir}/b{self.pred_batch_idx}_{num_graphs}.npz', sparse_matrix)
            self.pred_batch_idx += 1

        
        out = torch.split(torch.flatten(out), self.num_genes) #B, N
        out = torch.stack(out)

        ## uncertainty head
        if self.uncertainty:
            out_logvar = self.uncertainty_w(base_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
            return out, torch.stack(out_logvar)

        return out #B, N


class debug_model(torch.nn.Module):
    def __init__(self):
        super(debug_model, self).__init__()

    def forward(self, data):
        try:
            x, pert_idx = data.x, data.pert_idx
        except (KeyError, AttributeError):
            x, pert_idx = data.x, None
        num_graphs = len(data.batch.unique())    
        out = x[:, 0].reshape((num_graphs, -1))
        return out