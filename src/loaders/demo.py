# This script is implemented based on the https://github.com/Juanhui28/HeaRT
# Existing Train Settings: one negative sample per positive sample
# Existing Evaluation Settings: ? the same set of randomly sampled negatives are used for all positive samples. 
# for ogbl-citation2: randomly sample 1000 negative samples per positive sample due to scalable issue.

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
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
from transformers import BertTokenizer
from transformers import AutoTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME


import logging
import os, sys
sys.path.insert(0, '..')
from src.utilities.config import cfg, update_cfg
logger = logging.getLogger(__name__)

from kg_loader import KGProcessor, convert_examples_to_features

def main():
    
    cfg.gradient_accumulation_steps = 1
    cfg.no_cuda = True
    cfg.bert_model = cfg.lm.model.name 
    
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO if cfg.local_rank in [-1, 0] else logging.WARN)
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(cfg.local_rank)
        device = torch.device("cuda", cfg.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(cfg.local_rank)
        device = torch.device("cuda", cfg.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(cfg.local_rank != -1), cfg.fp16))

    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            cfg.gradient_accumulation_steps))
    
    # ------------------------------------------------------------------------ #
    # Model 
    # ------------------------------------------------------------------------ #
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm.model.name, do_lower_case=cfg.lm.do_lower_case)

    cache_dir = cfg.cache_dir if cfg.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(cfg.local_rank))
    model = BertForSequenceClassification.from_pretrained(cfg.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels)
    
    if cfg.fp16:
        model.half()
    model.to(device)
    if cfg.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        #model = torch.nn.parallel.data_parallel(model)

    # ------------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------------ #

    processors = {
        "kg": KGProcessor,
    }
    processor = processors[cfg.task_name]()
    
    label_list = processor.get_labels(cfg.data.dir)
    num_labels = len(label_list)
    logging.info("label_list: {}".format(label_list))
    entity_list = processor.get_entities(cfg.data.dir)
    #print(entity_list)
    train_examples = None
    num_train_optimization_steps = 0
    if cfg.do_train:
        train_examples = processor.get_train_examples(cfg.data.dir)
        num_train_optimization_steps = int(
            len(train_examples) / cfg.train_batch_size / cfg.gradient_accumulation_steps) * cfg.num_train_epochs
        if cfg.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if cfg.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, cfg.lm.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", cfg.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


if __name__ == "__main__":
    main()