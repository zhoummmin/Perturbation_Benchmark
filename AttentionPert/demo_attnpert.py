import sys
sys.path.append('../')
import re
import wandb
from attnpert import PertData, ATTNPERT_RECORD_TRAIN
from attnpert.model import *
from attnpert.utils import print_sys
import argparse
from attnpert.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis, non_zero_analysis

# Data names list
data_names = ['AdamsonWeissman2016_GSM2406675_1', 'AdamsonWeissman2016_GSM2406677_2', 'AdamsonWeissman2016_GSM2406681_3',
              'DatlingerBock2017_stimulated', 'DatlingerBock2017_unstimulated', 'DatlingerBock2021_stimulated', 'DatlingerBock2021_unstimulated',
              'Dixit_GSM2396858', 'Dixit_GSM2396861', 'PapalexiSatija2021_eccite_arrayed_RNA', 'PapalexiSatija2021_eccite_RNA',
              'TianKampmann2019_day7neuron',
              'TianKampmann2019_iPSC', 'TianKampmann2021_CRISPRa', 'TianKampmann2021_CRISPRi']

for data_name in data_names:
    print(f"-------{data_name}")
    data_lower = data_name.lower()
    gene2vec_path = f'/home/share/huadjyin/home/zhoumin3/AttentionPert/A_all_dataset/{data_lower}/gene2vec.npy'
    act_type = 'softmax'

    settings = {
        "gene2vec_args": {"gene2vec_file": gene2vec_path},
        "pert_local_min_weight": 0.75,
        "pert_local_conv_K": 1,
        "pert_weight_heads": 2,
        "pert_weight_head_dim": 64,
        "pert_weight_act": act_type,
        "non_add_beta": 5e-2,
        "record_pred": True
    }

    for seed in range(1, 6):
        pert_data = PertData('/home/share/huadjyin/home/zhoumin3/AttentionPert/A_all_dataset')
        pert_data.load(f'/home/share/huadjyin/home/zhoumin3/AttentionPert/A_all_dataset/{data_lower}')
        pert_data.prepare_split('simulation', seed)
        pert_data.get_dataloader(32, 64)

        attnpert_model = ATTNPERT_RECORD_TRAIN(pert_data, 'cuda', False, 'attnpert', f'{data_name}_split{seed}')
        attnpert_model.model_initialize(64, PL_PW_non_add_Model, f'{data_name}_split{seed}', **settings)
        print_sys(attnpert_model.config)

        save_path = f'/home/share/huadjyin/home/zhoumin3/zhoumin/model_benchmark/01_A_results/{data_name}/attnpert/split{seed}'
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving to {save_path}")

        attnpert_model.train(15, 1, save_path, data_name, seed)
        attnpert_model.save_model(save_path)

        print(f"split {seed} completed")
    print(f"{data_name}-------")
