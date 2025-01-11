import sys
from gears import PertData, GEARS
import scanpy as sc
from gears.utils import dataverse_download
from zipfile import ZipFile 
import json
import os
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import matplotlib
import wandb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

import pandas as pd
from scipy.stats import pearsonr

from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis, non_zero_analysis
# 
data_name = "AdamsonWeissman2016_GSM2406675_1"
# 
def evaluate_new(loader, model, uncertainty, device,  pert_data: PertData):
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
    

adata = sc.read_h5ad("/home/share/huadjyin/home/zhoumin3/zhoumin/benchmark_data/01A_total_re/03final/AdamsonWeissman2016_GSM2406675_1.h5ad")

pert_data = PertData('/home/share/huadjyin/home/zhoumin3/zhoumin/benchmark_data/01A_total_re/03final/', default_pert_graph=False)
pert_data.new_data_process(dataset_name='AdamsonWeissman2016_GSM2406675_1', adata=adata)

for seed in range(1, 6):
    pert_data.prepare_split(split='simulation', seed=seed)
    pert_data.get_dataloader(batch_size=64, test_batch_size=64)
    gears_model = GEARS(pert_data, device=device,
                        weight_bias_track=True,
                        proj_name='01_dataset_all_gears',
                        exp_name=f'{data_name}_split{seed}')
    gears_model.model_initialize(hidden_size=64)
    gears_model.train(epochs=15, lr=1e-3)

    model_save_path = f'/home/share/huadjyin/home/zhoumin3/zhoumin/model_benchmark/01_A_results/{data_name}/gears/split{seed}'
    os.makedirs(model_save_path, exist_ok=True)
    print(f"Saving to {model_save_path}")
    gears_model.save_model(model_save_path)

    print("---Creating test result")
    #test_result
    test_result = evaluate_new(gears_model.dataloader['test_loader'], gears_model.model, gears_model.config['uncertainty'], gears_model.device, pert_data)
    print("Test result saved successfully")

    results_to_save = [
        (test_result, f"{data_name}_split{seed}_test_result"),
        (compute_metrics(test_result), f"{data_name}_split{seed}_test_metrics"),
        (test_result, f"{data_name}_split{seed}_test_pert_result"),
        (deeper_analysis(pert_data.adata, test_result), f"{data_name}_split{seed}_deeper_result"),
        (non_dropout_analysis(pert_data.adata, test_result), f"{data_name}_split{seed}_non_dropout_result"),
        (non_zero_analysis(pert_data.adata, test_result), f"{data_name}_split{seed}_non_zero_result")
    ]

    # save results
    def save_results(result, file_prefix, save_path):
        file_path = f"{save_path}/{file_prefix}.pkl"
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            analysis_name = file_prefix.split('_')[0].replace('_', ' ').title()
            print(f"{analysis_name} result generation has an issue and needs to be regenerated")
            print(e)

    for result, file_prefix in results_to_save:
        save_results(result, file_prefix, model_save_path)

    wandb.finish()
    print(f"split {seed} computation completed")
