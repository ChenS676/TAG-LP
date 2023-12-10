
from __future__ import absolute_import, division, print_function
import logging
import os
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

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

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
from yacs.config import CfgNode as CN
# TODO update train script based on https://github.com/icmpnorequest/Pytorch_BERT_Text_Classification/blob/master/BERT_Text_Classification_CPU.ipynb
# Issue acc is 0
@timebudget
def train_loop(dataloader, 
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler, 
               num_labels, 
               num_train_optimization_steps, 
               global_step, 
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
        print("Training loss: ", tr_loss, nb_tr_examples)
    return global_step, nb_tr_steps, tr_loss
 
@timebudget
def eval_loop(eval_dataloader, 
              model, 
              num_labels, 
              tr_loss, 
              global_step, 
              cfg, 
              compute_metrics,
              nb_tr_steps, 
              device='cpu'):
    # eval_loop(eval_dataloader, model, device, num_labels, tr_loss, global_step, cfg, compute_metrics, nb_tr_steps)
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
    loss = tr_loss/nb_tr_steps if cfg.do_train else None

    logger.info("*** Example Eval ***")
    logger.info(f"preds: {preds[:10]}")
    logger.info(f"labels: {labels[0][:10]}")
    result['eval_loss'] = eval_loss
    result['global_step'] = global_step
    result['loss'] = loss

    output_eval_file = os.path.join(cfg.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def get_model(cfg: CN):

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(cfg.lm.model.name, do_lower_case=cfg.lm.do_lower_case)
    cache_dir = cfg.cache_dir if cfg.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(cfg.local_rank))
    model = BertForSequenceClassification.from_pretrained(cfg.bert_model,
              cache_dir=cache_dir,
              num_labels=cfg.num_labels)
    return model, tokenizer