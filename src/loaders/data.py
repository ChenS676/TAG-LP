import torch 
import logger 
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from timebudget import timebudget

@timebudget
def get_data(features, cfg, logger):
    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", cfg.train_batch_size)
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

