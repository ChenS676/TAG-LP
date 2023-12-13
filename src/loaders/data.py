import torch 
import logger 
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from timebudget import timebudget

@timebudget
def get_data(features, 
             cfg, 
             logger):
    
    # logger.info("***** Running *****")
    # logger.info("  Batch size = %d", cfg.train_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if cfg.local_rank == -1: # single GPU
        train_sampler = RandomSampler(data)
    else: # multi-GPU
        train_sampler = DistributedSampler(data)
    dataloader = DataLoader(data, 
                            sampler=train_sampler, 
                            batch_size=cfg.train_batch_size,
                            pin_memory=True,
                            shuffle=False
                            )
    return dataloader


def create_corrupt_list(test_triple: list, 
                        entity_list: list, 
                        mode: str,
                        all_triples_str_set: set):
    """create_corrupt_list through replace head and tail entity

    Args:
        test_triple (list): _description_
        entity_list (list): _description_
        mode (str): choice head or tail 
        all_triples_str_set (set): _description_

    Returns:
        _type_: _description_
    """
    head = test_triple[0]
    relation = test_triple[1]
    tail = test_triple[2]
    labels = []
    head_corrupt_list = [test_triple]
    labels.append(1)
    
    if mode == 'head':
        for corrupt_ent in entity_list:
            if corrupt_ent != head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(tmp_triple)
                    labels.append(0)
        return head_corrupt_list, labels
    
    elif mode == 'tail':
        tail_corrupt_list = [test_triple]
        for corrupt_ent in entity_list:
            if corrupt_ent != tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(tmp_triple)
                    labels.append(0)
        return tail_corrupt_list, labels
    


