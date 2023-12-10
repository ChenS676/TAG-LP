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


from IPython import embed
# TODO https://wandb.ai/wandb_fc/articles/reports/Monitor-Improve-GPU-Usage-for-Model-Training--Vmlldzo1NDQzNjM3#:~:text=Try%20increasing%20your%20batch%20size&text=Gradients%20for%20a%20batch%20are,increase%20the%20speed%20of%20calculation.
# improve GPU usage for model training
# python -m torch.distributed.launch loaders/demo.py --do_train
# TODO double check the evaluation method in KG, compare it with others 1. tangji liang lecture, 2. galkin new paper
# TODO add more evaluation metrics

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


from src.utilities.utils import compute_metrics, config_device
from src.utilities.utils import seed_everything, check_cfg, create_folders, ddp_setup
from src.loaders.data import get_data
from src.lm_trainer.bert_trainer import train_loop, eval_loop, get_model
from yacs.config import CfgNode as CN

                
@timebudget
def main_worker(gpu: int,
         ngpus_per_node: int,
         cfg: CN):
    cfg.gpu = gpu
    
    model, tokenizer = get_model(cfg)
    ddp_setup(cfg, ngpus_per_node, gpu, model)
    if cfg.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    
    # current device on local machine 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    processors = {
        "kg": KGProcessor,
    }
    processor = processors[cfg.task_name]()
    
    label_list = processor.get_labels(cfg.data.dir)
    num_labels = len(label_list)
    logging.info("label_list: {}".format(label_list))
    entity_list = processor.get_entities(cfg.data.dir)
    print(entity_list)
    # ------------------------------------------------------------------------ #
    # Model 
    # ------------------------------------------------------------------------ #
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
                  lr = 5e-2, # cfg.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # cfg.adam_epsilon  - default is 1e-8.
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
        #print(model)
        gloabl_step, nb_tr_steps, tr_loss = train_loop(dataloader, model, optimizer, scheduler, num_labels, num_train_optimization_steps, global_step, device)
                
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
    if cfg.do_eval and (cfg.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        eval_examples = processor.get_dev_examples(cfg.data.dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, cfg.lm.max_seq_length, tokenizer)
       
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(cfg.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(cfg.output_dir, do_lower_case=cfg.lm.do_lower_case)

        model.eval()

        eval_dataloader = get_data(eval_features, cfg, logger)
        eval_loop(eval_dataloader, model, num_labels, tr_loss, global_step, cfg, compute_metrics, nb_tr_steps, device)
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
    cfg.distributed = True
    cfg.total_steps = 1000  # Adjust the number of training steps
    cfg.warmup_steps = 100  # Adjust the number of warm-up steps
    cfg.num_labels = 2
    cfg.batch_size = cfg.lm.train.batch_size
    # ------------------------------------------------------------------------ #
    # data params for config
    # ------------------------------------------------------------------------ #

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        cfg.world_size = len(nodes)
        cfg.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        cfg.world_size = 1
        cfg.rank = 0
        nodes = ["127.0.0.1"]

    if cfg.distributed:
        mp.set_start_method('forkserver')
        print(cfg.rank)
        port = np.random.randint(15000, 15025)
        cfg.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(cfg.dist_url)
        cfg.dist_backend = 'nccl'
        cfg.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    cfg.num_workers = 11
    cfg.ngpus_per_node = ngpus_per_node

    if cfg.distributed:
        cfg.world_size = ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        if ngpus_per_node == 1:
            cfg.gpu = 0
        main_worker(cfg.gpu, ngpus_per_node, cfg)
        
    


