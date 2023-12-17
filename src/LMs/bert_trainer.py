from __future__ import absolute_import, division, print_function
import logging
import os
import sys
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss

from timebudget import timebudget
from IPython import embed
from yacs.config import CfgNode as CN
from sklearn import metrics

# Import local modules
sys.path.insert(0, '..')
from src.utilities.config import cfg, update_cfg
from src.utilities.utils import compute_metrics, config_device
from src.data_utils.kg_loader import convert_examples_to_features, get_eval_data

logger = logging.getLogger(__name__)
timebudget.set_quiet()  # Don't show measurements as they happen
timebudget.report_at_exit()  # Generate a report when the program exits


@timebudget
def train_loop(cfg: CN,
               dataloader, 
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler, 
               num_labels, 
               num_train_optimization_steps, 
               logger,
               device, 
               result):
    model.train()
    global_step = 0
    for _ in trange(int(cfg.lm.train.epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            optimizer.zero_grad()
            # define a new function to compute loss values for both output_modes
            outputs = model(input_ids, segment_ids, input_mask, labels=None)
            loss = outputs[0]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(loss.view(-1, num_labels), label_ids.view(-1))

            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

            if cfg.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
                # optimizer.step()
                
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if cfg.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = cfg.learning_rate * scheduler.get_lr(global_step/num_train_optimization_steps,
                                                                                cfg.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            logger.info(f"Training loss: {loss.item()}")
            avg_loss = tr_loss/nb_tr_steps if cfg.do_train else None
    logger.info(f"Average loss: {avg_loss}")
    result.update({'train_global_step': global_step})
    result.update({'train_avg_loss': avg_loss})
    return model, optimizer, scheduler
 
@timebudget
def eval_loop(
              mode,
              eval_dataloader, 
              model, 
              num_labels,
              device,
              result
              ):

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    labels = []
    model.to(device)
    model.eval()
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        # print(label_ids.view(-1))
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            labels.append(label_ids.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            labels[0] = np.append(
                labels[0], label_ids.detach().cpu().numpy(), axis=0)
            
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds[0], axis=1)
    result[f'{mode}_acc1'] = compute_metrics(preds, 
                                            labels[0])

    logger.info("*** Example Eval ***")
    logger.info(f"preds: {preds[:10]}")
    logger.info(f"labels: {labels[0][:10]}")
    result[f'{mode}_loss'] = eval_loss
    result[f'{mode}_acc2'] = metrics.accuracy_score(labels[0], preds)
    return result


def get_model(cfg: CN):

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm.model.name, do_lower_case=cfg.lm.do_lower_case)
    
    #cache_dir = cfg.cache_dir if cfg.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(cfg.local_rank))
    cache_dir = cfg.cache_dir if cfg.cache_dir else os.path.join('.', 'distributed_{}'.format(cfg.local_rank))
    model = BertForSequenceClassification.from_pretrained(cfg.bert_model,
              cache_dir=cache_dir,
              num_labels=cfg.num_labels)
    return model, tokenizer


def test_loop(processor, 
              model, 
              tokenizer,
              device):
        # hits@k mrr metrics
        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        top_ten_hit_count = 0
        all_triples_str_set, test_triples = processor.get_all_triples_str_set()
        label_list = processor.get_labels()
        entity_list = processor.get_entities()
        
        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])

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
                                        
    
            tmp_examples = processor._create_examples(head_corrupt_list, "test")
            print(len(tmp_examples))
            tmp_features = convert_examples_to_features('test', tmp_examples, label_list, cfg.lm.max_seq_length, tokenizer, print_info = False)
            eval_dataloader, all_label_ids = get_eval_data(tmp_features, cfg.eval_batch_size)
            model.eval()

            preds = []
            
            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)

                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)       

            preds = preds[0]
            # get the dimension corresponding to current label 1
            #print(preds, preds.shape)
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            #print(rel_values, rel_values.shape)
            _, argsort1 = torch.sort(rel_values, descending=True)
            #print(max_values)
            #print(argsort1)
            argsort1 = argsort1.cpu().numpy()
            rank1 = np.where(argsort1 == 0)[0][0]
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

            tmp_examples = processor._create_examples(tail_corrupt_list, "test")
            #print(len(tmp_examples))
            tmp_features = convert_examples_to_features('test', tmp_examples, label_list, cfg.lm.max_seq_length, tokenizer, print_info = False)
            eval_dataloader, _ = get_eval_data(tmp_features, cfg.eval_batch_size)
                        
            model.eval()
            preds = []        

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
            
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)

                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0) 

            preds = preds[0]
            # get the dimension corresponding to current label 1
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            _, argsort1 = torch.sort(rel_values, descending=True)
            argsort1 = argsort1.cpu().numpy()
            rank2 = np.where(argsort1 == 0)[0][0]
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)
            print('right: ', rank2)
            print('mean rank until now: ', np.mean(ranks))
            if rank2 < 10:
                top_ten_hit_count += 1
            print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

        return rank1, rank2, hits, hits_left, hits_right, ranks, ranks_left, ranks_right 

                    
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
                    
def create_optimizer(model, 
                     cfg):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if cfg.fp16:
        # TODO 
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

    else:
        optimizer = AdamW(model.parameters(),
                  lr = cfg.lr, # cfg.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # cfg.adam_epsilon  - default is 1e-8.
                )
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*cfg.num_train_optimization_steps), num_training_steps=cfg.num_train_optimization_steps)


    return scheduler, optimizer