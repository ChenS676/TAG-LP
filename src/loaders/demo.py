# This script is implemented based on the https://github.com/Juanhui28/HeaRT
# Existing Train Settings: one negative sample per positive sample
# Existing Evaluation Settings: ? the same set of randomly sampled negatives are used for all positive samples. 
# for ogbl-citation2: randomly sample 1000 negative samples per positive sample due to scalable issue.

from __future__ import absolute_import, division, print_function
import logging
import sys
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from kg_loader import KGProcessor, convert_examples_to_features
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, get_linear_schedule_with_warmup
import logging
import os, sys
sys.path.insert(0, '..')
from src.utilities.config import cfg, update_cfg
logger = logging.getLogger(__name__)
from timebudget import timebudget
timebudget.set_quiet()  # don't show measurements as they happen
timebudget.report_at_exit()  # Generate report when the program exits

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from IPython import embed
from src.utilities.utils import compute_metrics, config_device
from src.utilities.utils import seed_everything, check_cfg, create_folders, ddp_setup
from src.loaders.data import get_data
from src.lm_trainer.bert_trainer import train_loop, eval_loop
from yacs.config import CfgNode as CN


# TODO https://wandb.ai/wandb_fc/articles/reports/Monitor-Improve-GPU-Usage-for-Model-Training--Vmlldzo1NDQzNjM3#:~:text=Try%20increasing%20your%20batch%20size&text=Gradients%20for%20a%20batch%20are,increase%20the%20speed%20of%20calculation.
# improve GPU usage for model training
# python -m torch.distributed.launch loaders/demo.py --do_train
# TODO double check the evaluation method in KG, compare it with others 1. tangji liang lecture, 2. galkin new paper
# TODO add more evaluation metrics

@timebudget
def main(rank: int,
         world_size: int,
         cfg: CN):
    
    ddp_setup(rank, world_size)
    
    processors = {
        "kg": KGProcessor,
    }
    processor = processors[cfg.task_name]()
    
    label_list = processor.get_labels(cfg.data.dir)
    num_labels = len(label_list)
    logging.info("label_list: {}".format(label_list))
    entity_list = processor.get_entities(cfg.data.dir)
    #print(entity_list)
    # ------------------------------------------------------------------------ #
    # Model 
    # ------------------------------------------------------------------------ #
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm.model.name, do_lower_case=cfg.lm.do_lower_case)
    cache_dir = cfg.cache_dir if cfg.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(cfg.local_rank))
    model = BertForSequenceClassification.from_pretrained(cfg.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
    
    config_device(cfg, logger, model, rank)
    logger.info(f"model device {print(model.device)}, rank {rank}")
    seed_everything(cfg)
    check_cfg(cfg)
    create_folders(cfg)
    
    train_examples = None
    num_train_optimization_steps = 0
    if cfg.do_train:
        train_examples = processor.get_train_examples(cfg.data.dir)
        num_train_optimization_steps = int(
            len(train_examples) / cfg.train_batch_size / cfg.gradient_accumulation_steps) * cfg.lm.train.epochs
        if cfg.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
            
    cfg.train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if cfg.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=cfg.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if cfg.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=cfg.loss_scale)
            
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.total_steps)
    # BertAdam - Bert version of Adam algorithm with weight decay fix, warmup and linear decay of the learning rate.
    else:
        optimizer = AdamW(model.parameters(),
                  lr = 5e-2, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.total_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
        
    if cfg.do_train:
        features = convert_examples_to_features(
            train_examples, label_list, cfg.lm.max_seq_length, tokenizer)
        dataloader = get_data(features, cfg, logger)
        # ------------------------------------------------------------------------ #
        # Train
        # ------------------------------------------------------------------------ #
        model.train()
        model.to(rank)
        #print(model)
        gloabl_step, nb_tr_steps, tr_loss = train_loop(dataloader, model, optimizer, scheduler, rank, num_labels, num_train_optimization_steps, global_step)
                
    if cfg.do_train and (cfg.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(cfg.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(cfg.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(cfg.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(cfg.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(cfg.output_dir, do_lower_case=cfg.lm.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(cfg.bert_model, num_labels=num_labels)

    # ------------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------------ #
    model.to(rank)
    if cfg.do_eval and (cfg.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        eval_examples = processor.get_dev_examples(cfg.data.dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, cfg.lm.max_seq_length, tokenizer)
       
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(cfg.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(cfg.output_dir, do_lower_case=cfg.lm.do_lower_case)

        model.eval()

        eval_dataloader = get_data(eval_features, cfg, logger)
        eval_loop(eval_dataloader, model, rank, num_labels, tr_loss, global_step, cfg, compute_metrics, nb_tr_steps)
    destroy_process_group()

    
    
if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    # temporary solution for passing arguments
    cfg.gradient_accumulation_steps = 1
    cfg.no_cuda = False
    cfg.bert_model = cfg.lm.model.name 
    cfg.server_ip = ''
    cfg.server_port = ''
    cfg.local_rank = 0
    cfg.total_steps = 1000  # Adjust the number of training steps
    cfg.warmup_steps = 100  # Adjust the number of warm-up steps
    # ------------------------------------------------------------------------ #
    # data params for config
    # ------------------------------------------------------------------------ #

    world_size = torch.cuda.device_count()
    logger.info(f"world_size: {world_size}")
    mp.spawn(main, args=(world_size, cfg), nprocs=world_size)
    