import os
import time
import argparse
import pandas as pd
import scanpy as sc
from os.path import join as pjoin
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from gears_reduce.gears_ddp import GEARS
from gears_reduce import PertData
from gears_reduce.model import *
from gears_reduce.scfm_utils import *
from gears_reduce.utils import print_sys
import pandas as pd
from scipy.stats import pearsonr
from gears_reduce.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis, non_zero_analysis
import wandb


args = {
    'local_rank': 0,
    #'data_dir': '/home/ding.bai/pert_new/mywork/data',
    #'data_name': 'norman',
    'split': 'simulation',
    #'result_dir': './results',
    #'seed': 1,
    'epochs': 15,
    'batch_size': 32,
    'valid_every': 1,
    'test_batch_size': 64,
    'train_gene_set_size': 0.75,
    'hidden_size': 64,
    'device': 'cuda',
    'model_type': None,
    'bin_set': None,
    'singlecell_model_path': None,
    'finetune_method': None,
    'mode': 'v1',
    'accumulation_steps': 1,
    'wandb': False,
    'record_pred': False,
    'scfm_genes_list_path': "/home/share/huadjyin/home/zhoumin3/scLong/scLong/GEARS/selected_genes_27k.txt",
    'scfm_hyper_params_path': "/home/share/huadjyin/home/zhoumin3/scLong/scLong/GEARS/gocont_4096_48m_pretrain_1b_mix.pkl",
    'scfm_ckpt_path': "/home/share/huadjyin/home/zhoumin3/scLong/scLong/GEARS/gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth",
    'run_name': "default_run",
    'scfm_gene2vec_file': "/home/share/huadjyin/home/zhoumin3/scLong/scLong/GEARS/selected_gene2vec_27k.npy",
    'lr': 1e-3
}

local_rank = args["local_rank"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
if torch.cuda.is_available():# Use the device index (local_rank) to set the correct GPU device
    torch.cuda.set_device(local_rank)

is_master = 0


local_rank = int(os.environ.get('LOCAL_RANK', 0))
init_distributed_mode(local_rank)

def save_result(result, file_name, save_path):
    file_full_path = f"{save_path}/{file_name}.pkl"
    with open(file_full_path, 'wb') as f:
        pickle.dump(result, f)
        
data_names = ['AdamsonWeissman2016_GSM2406675_1', 'AdamsonWeissman2016_GSM2406677_2', 'AdamsonWeissman2016_GSM2406681_3']


for data_name in data_names:
    print(f"-------{data_name}")
    data = data_name.lower()

    for seed in range(1, 6):
        pert_data = PertData('/home/share/huadjyin/home/zhoumin3/scLong/A_all_dataset/') 
        pert_data.load(data_path = f'/home/share/huadjyin/home/zhoumin3/scLong/A_all_dataset/{data}') 
        input_genes_ens_ids = pert_data.adata.var.index.tolist()
        with open('/home/share/huadjyin/home/zhoumin3/scLong/scLong/GEARS/selected_genes_27k.txt', 'r') as f:
            scfm_genes_ens_ids = [line.rstrip('\n') for line in f.readlines()]
        if is_master:
            print("target and scfm genes intersect: ", len(np.intersect1d(input_genes_ens_ids, scfm_genes_ens_ids)))
            
        pert_data.prepare_split(split = 'simulation', seed = seed) 
        pert_data.get_dataloader(batch_size = 32, test_batch_size = 64) 
        gears_model = GEARS(pert_data,
                        local_rank=local_rank,
                        is_master=is_master,
                        world_size=1,
                        weight_bias_track=True,
                        train_bs=args['batch_size'],
                        test_bs=args['test_batch_size'],
                        device=device, 
                        proj_name = 'scLong', 
                        exp_name=f'{data_name}_split{seed}')
    
        gears_model.model_initialize(hidden_size=args['hidden_size'], 
                                     model_type=args['model_type'],
                                     bin_set=args['bin_set'],
                                     load_path=args['singlecell_model_path'],
                                     finetune_method=args['finetune_method'],
                                     accumulation_steps=args['accumulation_steps'],
                                     mode=args['mode'],
                                     input_genes_ens_ids=input_genes_ens_ids,
                                     scfm_genes_ens_ids=scfm_genes_ens_ids,
                                     scfm_hyper_params_path=args['scfm_hyper_params_path'],
                                     scfm_ckpt_path=args['scfm_ckpt_path'],
                                     scfm_class=None,  # This can be set to a specific class if available
                                     key_enc="merged_decodings",
                                     scfm_gene2vec_file=args['scfm_gene2vec_file'],
                                     record_pred=args['record_pred'])
            
        #gears_model.train(epochs = 15, lr = 1e-3, result_dir=args['result_dir'], lr=args['lr'], valid_every=valid_every)
    
        model_save_path = f'/home/share/huadjyin/home/zhoumin3/zhoumin/model_benchmark/01_A_results/{data_name}/sclong/split{seed}'
        os.makedirs(model_save_path, exist_ok=True)
        gears_model.train(epochs = 15, lr = 1e-3, result_dir=model_save_path, valid_every=1)
        print(f"saving to {model_save_path}")
        gears_model.save_model(model_save_path)
        
        # 
        print("---Creating test_res")
        test_res = evaluate(gears_model.dataloader['test_loader'], gears_model.model, gears_model.config['uncertainty'], gears_model.device)
        print("test_res saved successfully----")

        save_result(test_res, f"{data_name}_split{seed}_test_res", model_save_path)

        test_metrics, test_pert_res = compute_metrics(test_res)
        save_result(test_metrics, f"{data_name}_split{seed}_test_metrics", model_save_path)
        save_result(test_pert_res, f"{data_name}_split{seed}_test_pert_res", model_save_path)

        analyses = [
            ("deeper_analysis", "deeper_res"),
            ("non_dropout_analysis", "non_dropout_res"),
            ("non_zero_analysis", "non_zero_res")
        ]
        for analysis_func_name, result_file_name in analyses:
            try:
                analysis_func = globals()[analysis_func_name]
                result = analysis_func(pert_data.adata, test_res)
                save_result(result, f"{data_name}_split{seed}_{result_file_name}", model_save_path)
            except Exception as e:
                print(f"{analysis_func_name.replace('_', ' ').title()} result generation failed. Need to regenerate. Error: {e}")

        wandb.finish()
        print(f"split {seed} computation completed")

    print(f"{data_name}-------")
