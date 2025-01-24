import sys
import os

project_dir = os.path.abspath(os.path.join('..')) 
sys.path.append(project_dir)
import gears 
from gears import PertData, GEARS
from sklearn.model_selection import train_test_split
from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis, non_zero_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
import pytorch_lightning
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import scanpy as sc
import wandb

np.random.seed(42)
pytorch_lightning.seed_everything(42)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

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
    
pert_data = PertData('./data_GenePT', default_pert_graph=False) 
pert_data.load(data_path = './data_GenePT/adamsonweissman2016_gsm2406675_1')


for seed in range(1, 6):
    pert_data.prepare_split(split = 'simulation', seed = seed) 
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 64) 

    with open("./GenePT_gene_embeddings.pickle", "rb") as fp:
        GenePT_gene_embeddings = pickle.load(fp)
    gene_names= list(pert_data.adata.var['gene_name'].values)
    count_missing = 0
    EMBED_DIM = 1536 # embedding dim from GPT-3.5
    lookup_embed = np.zeros(shape=(len(gene_names),EMBED_DIM))
    for i, gene in enumerate(gene_names):
        if gene in GenePT_gene_embeddings:
            lookup_embed[i,:] = GenePT_gene_embeddings[gene].flatten()
        else:
            count_missing+=1
    
    gears_model = GEARS(pert_data, device = device, 
                            gene_emb = lookup_embed,
                            weight_bias_track = True, 
                            proj_name = 'scELMo_GenePT', 
                            exp_name = f'{data_name}_split{seed}')
    gears_model.model_initialize(hidden_size = 64)
    
    gears_model.train(epochs = 15, lr = 1e-3)

    model_save_path = f'/home/share/huadjyin/home/zhoumin3/zhoumin/model_benchmark/01_A_results/{data_name}/scELMo_GenePT/split{seed}'
    os.makedirs(model_save_path, exist_ok=True)
    print(f"saving to {model_save_path}")
    gears_model.save_model(model_save_path)
    
    # 
    print("---Creating test_res")
    test_res = evaluate_new(gears_model.dataloader['test_loader'], gears_model.model, gears_model.config['uncertainty'], gears_model.device, pert_data)
    print("test_res saved successfully----")
    
    results_info = [
        (test_res, f"{data_name}_split{seed}_test_res"),
        (compute_metrics(test_res), f"{data_name}_split{seed}_test_metrics"),
        (test_res, f"{data_name}_split{seed}_test_pert_res"),
        (deeper_analysis(pert_data.adata, test_res), f"{data_name}_split{seed}_deeper_res"),
        (non_dropout_analysis(pert_data.adata, test_res), f"{data_name}_split{seed}_non_dropout_res"),
        (non_zero_analysis(pert_data.adata, test_res), f"{data_name}_split{seed}_non_zero_res")
    ]

    for result, file_name in results_info:
        file_path = f"{model_save_path}/{file_name}.pkl"
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            analysis_name = file_name.split('_')[0].replace('_', ' ').title()
            print(f"{analysis_name} generation failed and needs to be regenerated. Error: {str(e)}")

    wandb.finish()
    print(f"split {seed} computation completed")
        
