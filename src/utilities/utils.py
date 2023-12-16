class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return  simple_accuracy(preds, labels)

import logging
import torch 
from yacs.config import CfgNode as CN

from timebudget import timebudget

def config_device(cfg: CN, 
                  logger: None, 
                  model: torch.nn.Module,
                  rank: int):   
    
    if cfg.server_ip and cfg.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(cfg.server_ip, cfg.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO if cfg.local_rank in [-1, 0] else logging.WARN)
    
    # multigpu
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
        cfg.n_gpu = torch.cuda.device_count()
        model.to(rank)
    else:
        torch.cuda.set_device(cfg.local_rank)
        device = torch.device("cuda", cfg.local_rank)
        cfg.n_gpu = 1
        model.to(device)
        
    if cfg.fp16:
        model.half()
   
    if cfg.local_rank != -1 and cfg.n_gpu > 1:
        model = DDP(model, device_ids=[rank])
        
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16))

import random
import numpy as np 

def seed_everything(cfg: CN):
    cfg.seed = random.randint(1, 200)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = False  
    
    torch.cuda.manual_seed_all(cfg.seed)
        
def check_cfg(cfg:CN):
    if (not cfg.do_train) and (not cfg.do_eval):
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            cfg.gradient_accumulation_steps))

import os 
def save_to_file(cfg: CN, 
                 result, 
                 logger, 
                 rank1, 
                 rank2,
                 hits,
                 hits_left,
                 hits_right,
                 ranks,
                 ranks_left,
                 ranks_right):
    import uuid 
    cfg.output_dir = os.path.join(cfg.output_dir, str(uuid.uuid4()))
    if os.path.exists(cfg.output_dir) and os.listdir(cfg.output_dir) and cfg.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(cfg.output_dir))
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    
    file_prefix  = str(cfg.train_batch_size) + "_" + str(cfg.lr) + "_" + str(cfg.lm.max_seq_length) + "_" + str(cfg.lm.train.epochs)
    #file_prefix = str(cfg.data.dir[7:])
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
            
        
# adapted from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

def ddp_setup(cfg, ngpus_per_node, gpu, model):
    """
    cfg:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    cfg.multigpu = False
    if cfg.distributed:
        # Use DDP
        cfg.multigpu = True
        cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        # cfg.batch_size = 8
        cfg.workers = int((cfg.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(cfg.gpu, cfg.rank, cfg.batch_size, cfg.workers)
        torch.cuda.set_device(cfg.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(cfg.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], output_device=cfg.gpu,
                                                          find_unused_parameters=True)

    elif cfg.gpu is None:
        # Use DP
        try:
            cfg.multigpu = True
            model = model.cuda()
            model = torch.nn.DataParallel(model)
        except:
            # current device on local machine 
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.to(device)
            
    if cfg.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    return cfg

def is_rank_zero(cfg):
    return cfg.rank == 0


