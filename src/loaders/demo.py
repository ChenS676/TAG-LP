# This script is implemented based on the https://github.com/Juanhui28/HeaRT
# Existing Train Settings: one negative sample per positive sample
# Existing Evaluation Settings: ? the same set of randomly sampled negatives are used for all positive samples. 
# for ogbl-citation2: randomly sample 1000 negative samples per positive sample due to scalable issue.

from __future__ import absolute_import, division, print_function
import logging
import sys
import numpy as np
import torch
import torch.multiprocessing as mp
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
import logging
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from yacs.config import CfgNode as CN

import os, sys
sys.path.insert(0, '..')
from src.utilities.config import cfg
logger = logging.getLogger(__name__)

from timebudget import timebudget
timebudget.set_quiet()  # don't show measurements as they happen
timebudget.report_at_exit()  # Generate report when the program exits

import warnings
warnings.simplefilter("ignore")
from IPython import embed
# TODO https://wandb.ai/wandb_fc/articles/reports/Monitor-Improve-GPU-Usage-for-Model-Training--Vmlldzo1NDQzNjM3#:~:text=Try%20increasing%20your%20batch%20size&text=Gradients%20for%20a%20batch%20are,increase%20the%20speed%20of%20calculation.
# improve GPU usage for model training
# python -m torch.distributed.launch loaders/demo.py --do_train
# TODO double check the evaluation method in KG, compare it with others 1. tangji liang lecture, 2. galkin new paper
# TODO add more evaluation metrics

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics

from src.utilities.utils import compute_metrics, config_device
from src.utilities.utils import seed_everything, check_cfg, create_folders, ddp_setup
from src.loaders.data import get_data
from src.lm_trainer.bert_trainer import train_loop, eval_loop, get_model
from src.loaders.kg_loader import KGProcessor, convert_examples_to_features


                
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
    
    if cfg.do_predict and (cfg.local_rank == -1 or torch.distributed.get_rank() == 0):

        train_triples = processor.get_train_triples(cfg.data.dir)
        dev_triples = processor.get_dev_triples(cfg.data.dir)
        test_triples = processor.get_test_triples(cfg.data.dir)
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)

        pred_examples = processor.get_test_examples(cfg.data.dir)
        pred_features = convert_examples_to_features(
            pred_examples, label_list, cfg.lm.max_seq_length, tokenizer)
        pred_dataloader = get_data(eval_features, cfg, logger)

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(cfg.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(cfg.output_dir, do_lower_case=cfg.lm.do_lower_case)
        
        model.to(device)
        model.eval()
        eval_loop(pred_dataloader, model, num_labels, tr_loss, global_step, cfg, compute_metrics, nb_tr_steps, device)

        # run link prediction
        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        top_ten_hit_count = 0

        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])

        '''
        file_prefix = str(cfg.data_dir[7:])
        f = open(file_prefix + '_ranks.txt','r')
        lines = f.readlines()
        for line in lines:
            temp = line.strip().split()
            rank1 = int(temp[0])
            ranks_left.append(rank1+1)
            print('left: ', rank1)
            ranks.append(rank1+1)
            if rank1 < 10:
                top_ten_hit_count += 1
            rank2 = int(temp[1])
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)
            print('right: ', rank2)
            print('mean rank until now: ', np.mean(ranks))
            if rank2 < 10:
                top_ten_hit_count += 1
            print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))                
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)
    
        '''
        for test_triple in test_triples:
            head = test_triple[0]
            relation = test_triple[1]
            tail = test_triple[2]
            #print(test_triple, head, relation, tail)

            head_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != head:
                    tmp_triple = [corrupt_ent, relation, tail]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        head_corrupt_list.append(tmp_triple)

            tmp_examples = processor._create_examples(head_corrupt_list, "test", cfg.data.dir)
            print(len(tmp_examples))
            tmp_features = convert_examples_to_features(tmp_examples, label_list, cfg.lm.max_seq_length, tokenizer, print_info = False)
            test_dataloader = get_data(tmp_features, cfg, logger)
            model.eval()

            preds = []
            labels = []
            for input_ids, input_mask, segment_ids, label_ids in test_dataloader:

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)
                    labels.append(label_ids.detach().cpu().numpy())
                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)      
                    labels[0] = np.append(labels.detach().cpu().numpy()[0], label_ids, axis=0)

            preds = preds[0]
            # get the dimension corresponding to current label 1
            #print(preds, preds.shape)
            rel_values = preds[:, labels[0]]
            rel_values = torch.tensor(rel_values)
            #print(rel_values, rel_values.shape)
            _, cfgort1 = torch.sort(rel_values, descending=True)
            #print(max_values)
            #print(cfgort1)
            cfgort1 = cfgort1.cpu().numpy()
            rank1 = np.where(cfgort1 == 0)[0][0]
            print('left: ', rank1)
            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            if rank1 < 10:
                top_ten_hit_count += 1

            tail_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != tail:
                    tmp_triple = [head, relation, corrupt_ent]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        tail_corrupt_list.append(tmp_triple)

            tmp_examples = processor._create_examples(tail_corrupt_list, "test", cfg.data.dir)
            #print(len(tmp_examples))
            tmp_features = convert_examples_to_features(tmp_examples, label_list, cfg.lm.max_seq_length, tokenizer, print_info = False)
            test_dataloader = get_data(tmp_features, cfg, logger)
            model.eval()
            preds = []        

            for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
            
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)
                    labels.append(label_ids.detach().cpu().numpy())
                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)      
                    labels[0] = np.append(labels.detach().cpu().numpy()[0], label_ids, axis=0)


            preds = preds[0]
            # get the dimension corresponding to current label 1
            rel_values = preds[:, labels[0]]
            rel_values = torch.tensor(rel_values)
            _, cfgort1 = torch.sort(rel_values, descending=True)
            cfgort1 = cfgort1.cpu().numpy()
            rank2 = np.where(cfgort1 == 0)[0][0]
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)
            print('right: ', rank2)
            print('mean rank until now: ', np.mean(ranks))
            if rank2 < 10:
                top_ten_hit_count += 1
            print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

            file_prefix = str(cfg.data.dir[7:]) + "_" + str(cfg.train_batch_size) + "_" + str(cfg.lm.learning_rate) + "_" + str(cfg.lm.max_seq_length) + "_" + str(cfg.lm.train.epochs)

            f = open(file_prefix + '_ranks.txt','a')
            f.write(str(rank1) + '\t' + str(rank2) + '\n')
            f.close()
            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)
    

        for i in [0,2,9]:
            logger.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
            logger.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
            logger.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
        logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))            

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
    cfg.gpu = None
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
        
    


