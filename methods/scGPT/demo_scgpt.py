import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings
import pickle
import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis, non_zero_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id
import wandb

# 
matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

set_seed(42)

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 2 

n_hvg = 0  # number of highly variable genes
include_zero_gene = "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
max_seq_len = 1536
# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
amp = True

load_model = "/home/share/huadjyin/home/zhoumin3/zhoumin/scgpt/scGPT_human"
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]
# settings for optimizer
lr = 1e-4  # or 1e-4
#batch_size = 64
batch_size = 32
#eval_batch_size = 64
eval_batch_size = 32
epochs = 15
schedule_interval = 1
early_stop = 5

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100

split = "simulation"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device)
                else:  # when batch-wise
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)
#
def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device, pert_data: PertData
) -> Dict:
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
    pred_gene_ids = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(batch, include_zero_gene, gene_ids=gene_ids)
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
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)
    results["pred_gene_ids"] = [pert_data.adata.var_names[i] for i in range(len(gene_ids))]
    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)
    results["gene_ids_de"] = gene_ids_de

    return results

data_name = "AdamsonWeissman2016_GSM2406675_1"

#import scanpy as sc
#adata = sc.read_h5ad("/home/share/huadjyin/home/zhoumin3/zhoumin/benchmark_data/01A_total_re/03scgpt/AdamsonWeissman2016_GSM2406675_1.h5ad")
#pert_data = PertData('/home/share/huadjyin/home/zhoumin3/zhoumin/benchmark_data/01A_total_re/03scgpt/')
#pert_data.new_data_process(dataset_name = data_name, adata = adata)

for seed in range(2, 6):
    save_dir = Path(f"/home/share/huadjyin/home/zhoumin3/zhoumin/model_benchmark/01_A_results/{data_name}/scgpt/split{seed}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving to {save_dir}")
    
    wandb.init(project='01_dataset_all_scgpt', name=f'{data_name}_split{seed}', config={
        "seed": seed,
        "data_name": data_name
    })
    
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    # log running date and current git commit
    run_time = time.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Running on {run_time}")
    wandb.log({"run_time": run_time})

    pert_data = PertData('/home/share/huadjyin/home/zhoumin3/zhoumin/benchmark_data/01A_total_re/03scgpt/') 
    pert_data.load(data_path = '/home/share/huadjyin/home/zhoumin3/zhoumin/benchmark_data/01A_total_re/03scgpt/adamsonweissman2016_gsm2406675_1')
    pert_data.prepare_split(split=split, seed=seed)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
    
    if load_model is not None:
        model_dir = Path(load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"
    
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
    
        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        match_info = f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}."
        logger.info(match_info)
        wandb.log({"match_info": match_info})
        genes = pert_data.adata.var["gene_name"].tolist()
    
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        resume_model_info = (
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        logger.info(resume_model_info)
        wandb.log({"resume_model_info": resume_model_info})
        
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)
    
    ntokens = len(vocab)  # size of vocabulary
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        do_mvc=MVC,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        use_fast_transformer=use_fast_transformer,
    )
    if load_param_prefixs is not None and load_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if any([k.startswith(prefix) for prefix in load_param_prefixs])
        }
        for k, v in pretrained_dict.items():
            loading_params_info = f"Loading params {k} with shape {v.shape}"
            logger.info(loading_params_info)
            wandb.log({"loading_params_info": loading_params_info})
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    elif load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            loading_all_params_info = f"Loading all model params from {model_file}"
            logger.info(loading_all_params_info)
            wandb.log({"loading_all_params_info": loading_all_params_info})
        except:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                loading_params_info = f"Loading params {k} with shape {v.shape}"
                logger.info(loading_params_info)
                wandb.log({"loading_params_info": loading_params_info})
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    model.to(device)
    
    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    best_val_loss = float("inf")
    best_model = None
    patience = 0
        
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]
    
        train(
            model,
            train_loader,
        )
        val_loss, val_mre = evaluate(
            model,
            valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        epoch_log = f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss/mse {val_loss:5.4f} |"
        logger.info("-" * 89)
        logger.info(epoch_log)
        logger.info("-" * 89)
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_mre": val_mre, "elapsed_time": elapsed})
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_info = f"Best model with score {best_val_loss:5.4f}"
            logger.info(best_model_info)
            wandb.log({"best_model_info": best_model_info})
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                early_stop_info = f"Early stop at epoch {epoch}"
                logger.info(early_stop_info)
                wandb.log({"early_stop_info": early_stop_info})
                break
    
        torch.save(
            model.state_dict(),
            save_dir / f"model_{epoch}.pt",
        )
    
        scheduler.step()
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    
    test_loader = pert_data.dataloader["test_loader"]
    print("---Creating test_res")
    test_res = eval_perturb(test_loader, best_model, device, pert_data)
    print("test_res saved successfully----")
    #wandb.log({"test_res": test_res})

    def save_result(result, file_name, save_dir):
        file_path = f"{save_dir}/{file_name}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

    save_result(test_res, f"{data_name}_split{seed}_test_res", save_dir)

    test_metrics, test_pert_res = compute_metrics(test_res)
    save_result(test_metrics, f"{data_name}_split{seed}_test_metrics", save_dir)
    wandb.log({"test_metrics": test_metrics})
    save_result(test_pert_res, f"{data_name}_split{seed}_test_pert_res", save_dir)

    for analysis_func, result_name in [
        (deeper_analysis, "deeper_res"),
        (non_dropout_analysis, "non_dropout_res"),
        (non_zero_analysis, "non_zero_res")
    ]:
        try:
            result = analysis_func(pert_data.adata, test_res)
            save_result(result, f"{data_name}_split{seed}_{result_name}", save_dir)
            #wandb.log({result_name: result})
        except Exception as e:
            print(f"{result_name} generation failed. Regeneration required")
            print(e)

    print(f"split{seed} computation completed")
    wandb.finish()
