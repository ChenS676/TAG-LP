import torch 
import logger 
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from timebudget import timebudget

@timebudget
def get_train_data(features, 
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
    return dataloader, all_label_ids

def get_eval_data(eval_features, eval_batch_size):

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)


    return eval_dataloader, all_label_ids



