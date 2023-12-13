from __future__ import absolute_import, division, print_function
import logging
import os
import sys
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertTokenizer, BertForSequenceClassification
from timebudget import timebudget
from IPython import embed
from yacs.config import CfgNode as CN
from sklearn import metrics

# Import local modules
sys.path.insert(0, '..')
from src.utilities.config import cfg, update_cfg

logger = logging.getLogger(__name__)
timebudget.set_quiet()  # Don't show measurements as they happen
timebudget.report_at_exit()  # Generate a report when the program exits
from pdb import set_trace as stop
# TODO: Accelerate train
# TODO: Test on CPU

@timebudget
def train_loop(cfg: CN,
               dataloader, 
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler, 
               num_labels, 
               num_train_optimization_steps, 
               global_step, 
               logger,
               device):
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
    return model, optimizer, scheduler, global_step
 
@timebudget
def eval_loop(eval_dataloader, 
              model, 
              num_labels, 
              global_step, 
              cfg, 
              compute_metrics,
              device='cpu'):

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    labels = []
    model.to(device)
    
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
    preds = preds[0]

    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, labels[0])

    logger.info("*** Example Eval ***")
    logger.info(f"preds: {preds[:10]}")
    logger.info(f"labels: {labels[0][:10]}")
    result['eval_loss'] = eval_loss
    result['global_step'] = global_step

    output_eval_file = os.path.join(cfg.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    logger.info("Triple classification acc is : ")
    logger.info(f"labels shape: {labels[0].shape}, preds shape: {preds.shape}")
    logger.info(f"acc score: {metrics.accuracy_score(labels[0], preds)}")


def get_model(cfg: CN):

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm.model.name, do_lower_case=cfg.lm.do_lower_case)
    cache_dir = cfg.cache_dir if cfg.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(cfg.local_rank))
    model = BertForSequenceClassification.from_pretrained(cfg.bert_model,
              cache_dir=cache_dir,
              num_labels=cfg.num_labels)
    return model, tokenizer


def test_loop_left(test_dataloader, model, device, ranks, ranks_left, top_ten_hit_count):

    preds = []
    labels = []
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
        
        stop()
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            labels.append(label_ids.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            labels[0] = np.append(
                labels[0], label_ids.detach().cpu().numpy(), axis=0)

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
    ranks.append(rank1+1)
    ranks_left.append(rank1+1)
    if rank1 < 10:
        top_ten_hit_count += 1
        
    return ranks, ranks_left, rank1, top_ten_hit_count


def test_loop_right(test_dataloader, model, device, ranks, ranks_right, top_ten_hit_count):
    preds = []        
    labels = []
    for i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(test_dataloader):
    
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)[0]

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            labels.append(label_ids.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            labels[0] = np.append(
                labels[0], label_ids.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    # get the dimension corresponding to current label 1
    rel_values = preds[:, labels[0]]
    rel_values = torch.tensor(rel_values)
    _, cfgort1 = torch.sort(rel_values, descending=True)
    cfgort1 = cfgort1.cpu().numpy()
    rank2 = np.where(cfgort1 == 0)[0][0]
    ranks.append(rank2+1)
    ranks_right.append(rank2+1)
    logger.info(f'right: {rank2}')
    logger.info(f'mean rank until now: {np.mean(ranks)}')
    if rank2 < 10:
        top_ten_hit_count += 1
    logger.info(f"hit@10 until now: {top_ten_hit_count * 1.0 / len(ranks)}")
    
    return ranks, ranks_right, rank2, top_ten_hit_count