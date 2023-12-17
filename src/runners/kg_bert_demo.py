# This script is implemented based on the https://github.com/Juanhui28/HeaRT
# and https://github.com/yao8839836/kg-bert/blob/master/run_bert_link_prediction.py
# Existing Train Settings: one negative sample per positive sample
# Existing Evaluation Settings: ? the same set of randomly sampled negatives are used for all positive samples. 
# for ogbl-citation2: randomly sample 1000 negative samples per positive sample due to scalable issue.

from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from yacs.config import CfgNode as CN

import os, sys
sys.path.insert(0, '..')
from src.utilities.config import cfg
logger = logging.getLogger(__name__)

from timebudget import timebudget
timebudget.set_quiet() 
timebudget.report_at_exit() 

import warnings
warnings.simplefilter("ignore")
from IPython import embed


from src.utilities.utils import seed_everything, check_cfg, save_to_file, ddp_setup
from src.data_utils.kg_loader import get_train_data, get_eval_data
from src.LMs.bert_trainer import train_loop, eval_loop, get_model, create_optimizer, test_loop
from src.data_utils.kg_loader import KGProcessor, convert_examples_to_features

# for debug
from pdb import set_trace as stop

                
@timebudget
def main_worker(gpu: int,
         ngpus_per_node: int,
         cfg: CN):
    cfg.gpu = gpu
    
    model, tokenizer = get_model(cfg)
    ddp_setup(cfg, ngpus_per_node, gpu, model)
    
    # current device on local machine 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    processors = {
        "kg": KGProcessor,
    }
    
    processor = processors[cfg.task_name](cfg.data.dir)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logging.info("label_list: {}".format(label_list))
    entity_list = processor.get_entities()
    print(entity_list)
    # ------------------------------------------------------------------------ #
    # Model 
    # ------------------------------------------------------------------------ #
    seed_everything(cfg)
    check_cfg(cfg)
    train_examples = None
    num_train_optimization_steps = 0
    if cfg.do_train:
        train_examples = processor.get_train_examples()
        cfg.num_train_optimization_steps = int(
            len(train_examples) / cfg.train_batch_size / cfg.gradient_accumulation_steps) * cfg.lm.train.epochs
        if cfg.local_rank != -1:
            cfg.num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
            
    cfg.train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    scheduler, optimizer = create_optimizer(model, cfg)
    result = {}
    
    if cfg.do_train:
        features = convert_examples_to_features(
            'train', train_examples, label_list, cfg.lm.max_seq_length, tokenizer)
        dataloader, _ = get_train_data(features, cfg, logger)
        # ------------------------------------------------------------------------ #
        # Train
        # ------------------------------------------------------------------------ #
        model.train()
        model, optimizer, scheduler = train_loop(cfg, 
                                                 dataloader, 
                                                 model, 
                                                 optimizer, 
                                                 scheduler, 
                                                 num_labels, 
                                                 num_train_optimization_steps,
                                                 logger, 
                                                 device, 
                                                 result)
        
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
        eval_examples = processor.get_dev_examples()
        eval_features = convert_examples_to_features('eval', 
            eval_examples, label_list, cfg.lm.max_seq_length, tokenizer)
       
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(cfg.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(cfg.output_dir, do_lower_case=cfg.lm.do_lower_case)

        model.eval()

        eval_dataloader, _ = get_eval_data(eval_features, cfg.eval_batch_size)
        result = eval_loop('eval', eval_dataloader, model, num_labels, device, result)
    
    if cfg.do_predict and (cfg.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(cfg.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(cfg.output_dir, do_lower_case=cfg.lm.do_lower_case)
        
        model.to(device)
        model.eval()

        pred_examples = processor.get_test_examples()
        pred_features = convert_examples_to_features('eval', 
            pred_examples, label_list, cfg.lm.max_seq_length, tokenizer)
        pred_dataloader, _ = get_eval_data(pred_features, cfg.eval_batch_size)

        result = eval_loop('test', pred_dataloader, model, num_labels, device, result)
        print(result)
        
        rank1, rank2, hits, hits_left, hits_right, ranks, ranks_left, ranks_right = test_loop(
                                                                                            processor, 
                                                                                            model,
                                                                                            tokenizer, 
                                                                                            device)

    save_to_file(cfg, 
                 result, 
                 logger, 
                 rank1, 
                 rank2, 
                 hits, 
                 hits_left, 
                 hits_right, 
                 ranks, 
                 ranks_left, 
                 ranks_right)
    output_eval_file = os.path.join(cfg.output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    
    if torch.distributed.is_initialized():
        destroy_process_group()

    
   
 
if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    # temporary solution for passing arguments

    cfg.no_cuda = False
    cfg.bert_model = 'bert-base-uncased'
    cfg.lm.max_seq_length = 15
    cfg.server_ip = ''
    cfg.server_port = ''
    cfg.local_rank = -1 # 0 distributed 
    cfg.distributed = False # True distributed
        
    cfg.num_labels = 2 
    cfg.gpu = None # None distributed
    
    cfg.do_train = True
    cfg.do_eval = True
    cfg.do_predict = True
    cfg.train_batch_size = 32
    cfg.gradient_accumulation_steps = 1
  
    cfg.eval_batch_size = 135
    cfg.lm.train.epochs = 5
    cfg.lr = 5e-5



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
        
    


