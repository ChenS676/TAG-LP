
from __future__ import absolute_import, division, print_function
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
from src.utilities.utils import AverageMeter

logger = logging.getLogger(__name__)
from timebudget import timebudget
from pdb import set_trace as stop

ROOT_PATH = '/pfs/work7/workspace/scratch/cc7738-prefeature1/TAG-LP/data'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        cfg:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod 
    @timebudget 
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        logging.info(f"load {input_file} tsv.")
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in tqdm(reader):
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                # print(line)
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self, dir):
        self.labels = set()
        self.dir = dir 
        self.get_rel2text()
        self.get_ent2text()
        self.entities = self.get_entities()
        self.relations = self.get_relations()
        self.all_triples_str_set, self.test_triples = self.get_all_triples_str_set()

    
    def get_rel2text(self):
        self.rel2text = {}
        with open(os.path.join(ROOT_PATH, f"{self.dir}/relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                self.rel2text[temp[0]] = temp[1]      
    
    
    def get_ent2text(self):
        self.ent2text = {}
        with open(os.path.join(ROOT_PATH, f"{self.dir}/entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]#.find(',')
                    self.ent2text[temp[0]] = temp[1]#[:end]
                    
        if self.dir.find("FB15") != -1:
            with open(os.path.join(ROOT_PATH, f"{self.dir}/entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    #first_sent_end_position = temp[1].find(".")
                    self.ent2text[temp[0]] = temp[1]#[:first_sent_end_position + 1]

    

    def get_entities(self):
        return list(self.ent2text.keys())
                    
    # _create_examples(self, lines, set_type):
    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self.get_train_triples(), "train")


    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self.get_dev_triples(), "dev")


    def get_test_examples(self):
      """See base class."""
      return self._create_examples(
          self.get_test_triples(), "test")


    def get_train_triples(self):
        """Gets training triples."""
        return self._read_tsv(ROOT_PATH + '/'+ self.dir + '/' + "train.tsv")


    def get_dev_triples(self):
        """Gets validation triples."""
        return self._read_tsv(ROOT_PATH + '/'+ self.dir + '/' + "dev.tsv")


    def get_test_triples(self):
        """Gets test triples."""
        return self._read_tsv(ROOT_PATH + '/'+ self.dir + '/' + "test.tsv")


    def get_relations(self):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(ROOT_PATH+'/'+ self.dir + '/' + "relations.txt", 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations


    def get_labels(self):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]


    def get_entities(self):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels) 
        with open(os.path.join(ROOT_PATH, f"{self.dir}/entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in tqdm(lines):
                # print(line.strip())
                entities.append(line.strip())
        return entities
    
    @timebudget
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        lines_str_set = set(['\t'.join(line) for line in lines])
        
        examples = []
        for (i, line) in enumerate(lines):
            head_ent_text = self.ent2text[line[0]]
            tail_ent_text = self.ent2text[line[2]]
            relation_text = self.rel2text[line[1]]
            
            if (set_type == "dev" ):
                label = "1"
                guid = "%s-%s" % (set_type, i)
                # print(guid)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))
                
            elif set_type == "test":
                label = "1" if i == 0 else "0"
                guid = "%s-%s" % (set_type, i)
                # print(guid)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))
            
                
            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                # print(guid)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))


                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # print(guid)
                    # corrupting head
                    for j in range(5):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(self.entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break                    
                        tmp_head_text = self.ent2text[tmp_head]
                        examples.append(
                            InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c = text_c, label="0"))       
                else:
                    # corrupting tail
                    tmp_tail = ''
                    # print(guid)
                    for j in range(5):
                        while True:
                            tmp_ent_list = set(self.entities)
                            tmp_ent_list.remove(line[2])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = self.ent2text[tmp_tail]
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail_text, label="0"))                                                  
        return examples

    def get_all_triples_str_set(self):

        train_triples = self.get_train_triples()
        dev_triples = self.get_dev_triples()
        test_triples = self.get_test_triples()
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)
        return all_triples_str_set, test_triples
    
    
@timebudget
def convert_examples_to_features(mode, examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None
        avg_len = AverageMeter()
        
        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
            #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            avg_len = _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4, avg_len)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)        

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        # logger.info(f"*** Example {mode}***")
        # logger.info("number of examples: %d" % (len(examples)))
        if ex_index < 5 and print_info:
            logger.info(f"guid: {example.guid}, label: {label_id}")
            # logger.info("tokens: %s" % " ".join(
            #        [str(x) for x in tokens]))
            # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            # logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(f"*** *** *** ***")

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    # logger.info("avg_len: {}".format(avg_len.avg))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
            
def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length, avg_len):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        avg_len.update(total_length)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()
            
    return avg_len
    
