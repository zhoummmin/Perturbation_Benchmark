import os
import time
import argparse
import pandas as pd
import scanpy as sc
from os.path import join as pjoin

from gears import PertData, GEARS
from datetime import datetime

import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis, non_zero_analysis
import pickle
import torch
import wandb

model_type='maeautobin'
bin_set='autobin_resolution_append'
finetune_method='frozen'
singlecell_model_path='/home/share/huadjyin/home/zhoumin3/zhoumin/scfoundation/scFoundation/model/models/models.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

def evaluate_new(loader, model, uncertainty, device, pert_data: PertData):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    gene_ids_de = []
    results = {}
    logvar = []
    
    for itr, batch in enumerate(loader):

        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            if uncertainty:
                p, unc = model(batch)
                logvar.extend(unc.cpu())
            else:
                p = model(batch)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            
            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])
                gene_names_de = [pert_data.adata.var_names[i] for i in de_idx]
                gene_ids_de.append(gene_names_de)

    # all genes
    results['pert_cat'] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()
    #results["pred_gene_ids"] = [pert_data.adata.var_names[i] for i in range(len(gene_ids))]

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results['pred_de']= pred_de.detach().cpu().numpy()
    results['truth_de']= truth_de.detach().cpu().numpy()
    results["gene_ids_de"] = gene_ids_de
    
    if uncertainty:
        results['logvar'] = torch.stack(logvar).detach().cpu().numpy()
    
    return results
#
data_name = "AdamsonWeissman2016_GSM2406681_3"
#
base_args = {
    #'device_id': '0',
    'data_dir': '/home/share/huadjyin/home/zhoumin3/zhoumin/scfoundation/scFoundation/GEARS/01_re_dataset_all_data/',
    'data_name': data_name,
    'split': 'simulation',
    'result_dir': '/home/share/huadjyin/home/zhoumin3/zhoumin/model_benchmark/01_A_results/',

    'epochs': 15,
    'batch_size': 4,
    'accumulation_steps': 5,
    'test_batch_size': 4,
    'hidden_size': 512,
    'train_gene_set_size': 0.75,
    'mode': 'v1',
    'highres': 0,
    'lr': 1e-3,
    'device': 'cuda',
    'model_type': 'maeautobin',
    'bin_set': 'autobin_resolution_append',
    'finetune_method': 'frozen',
    'singlecell_model_path': '/home/share/huadjyin/home/zhoumin3/zhoumin/scfoundation/scFoundation/model/models/models.ckpt'
}

pert_data = PertData(base_args['data_dir'])
#adata = sc.read_h5ad(pjoin(base_args['data_dir'], base_args['data_name']+'.h5ad'))
#sc.pp.normalize_total(adata)
#sc.pp.log1p(adata)
#adata.uns['log1p'] = {}
#adata.uns['log1p']['base'] = None
#pert_data.new_data_process(dataset_name=base_args['data_name'], adata=adata)
pert_data.load(data_path = '/home/share/huadjyin/home/zhoumin3/zhoumin/scfoundation/scFoundation/GEARS/01_re_dataset_all_data/adamsonweissman2016_gsm2406681_3') 

for seed in range(5, 6):
    args = base_args.copy() 
    args['seed'] = seed
    args['result_dir'] = os.path.join(args['result_dir'], f"{args['data_name']}/scfoundation/split{args['seed']}")
    os.makedirs(args['result_dir'], exist_ok=True)
    
    pert_data.prepare_split(split = args['split'], seed = args['seed'], train_gene_set_size=args['train_gene_set_size']) 
    pert_data.get_dataloader(batch_size = args['batch_size'], test_batch_size = args['test_batch_size']) 
    
    gears_model = GEARS(pert_data, device = device, 
                            weight_bias_track = True, 
                            proj_name = '01_dataset_all_scfoundation', 
                            exp_name = f'{data_name}_split{seed}')
    gears_model.model_initialize(hidden_size = args['hidden_size'], 
                                 model_type = args['model_type'],
                                 bin_set=args['bin_set'],
                                 load_path=args['singlecell_model_path'],
                                 finetune_method=args['finetune_method'],
                                 accumulation_steps=args['accumulation_steps'],
                                 mode=args['mode'],
                                 highres=args['highres'])
    gears_model.train(epochs = args['epochs'], result_dir=args['result_dir'], lr=args['lr'])
    print(f"saving to {args['result_dir']}")
    gears_model.save_model(args['result_dir'])

    param_pd = pd.DataFrame(args, index=['params']).T
    param_pd.to_csv('{}/params.csv'.format(args['result_dir'])) # zhoumin
    print("---Creating test_res")
    test_res = evaluate_new(gears_model.dataloader['test_loader'], gears_model.model, gears_model.config['uncertainty'], gears_model.device, pert_data)
    print("test_res saved successfully----")
    
    results_to_save = [
        (test_res, f"{args['data_name']}_split{args['seed']}_test_res"),
        (compute_metrics(test_res), f"{args['data_name']}_split{args['seed']}_test_metrics"),
        (test_pert_res, f"{args['data_name']}_split{args['seed']}_test_pert_res"),
        (deeper_analysis(pert_data.adata, test_res), f"{args['data_name']}_split{args['seed']}_deeper_res"),
        (non_dropout_analysis(pert_data.adata, test_res), f"{args['data_name']}_split{args['seed']}_non_dropout_res"),
        (non_zero_analysis(pert_data.adata, test_res), f"{args['data_name']}_split{args['seed']}_non_zero_res")
    ]

    for result, file_name in results_to_save:
        file_path = f"{args['result_dir']}/{file_name}.pkl"
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"{file_name.split('_')[0].title()} generation failed. Regeneration required. Error: {e}")

    wandb.finish()
    print(f"Split {seed} computation completed")
    

    


    
    
