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
    return {"acc": simple_accuracy(preds, labels)}

import logging
import torch 
from yacs.config import CfgNode as CN

from timebudget import timebudget
from IPython import embed

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
    else:
        torch.cuda.set_device(cfg.local_rank)
        device = torch.device("cuda", cfg.local_rank)
        cfg.n_gpu = 1
        
    if cfg.fp16:
        model.half()
    model.to(device)
    if cfg.local_rank != -1 and cfg.n_gpu > 1:
        model = DDP(model, device_ids=[rank])
        
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16))
    return device

import random
import numpy as np 

def seed_everything(cfg: CN):
    cfg.seed = random.randint(1, 200)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = False  
    
    if cfg.n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.seed)
        
def check_cfg(cfg:CN):
    if not cfg.do_train and not cfg.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            cfg.gradient_accumulation_steps))

import os 
def create_folders(cfg: CN):
    import uuid 
    cfg.output_dir = os.path.join(cfg.output_dir, str(uuid.uuid4()))
    if os.path.exists(cfg.output_dir) and os.listdir(cfg.output_dir) and cfg.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(cfg.output_dir))
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
        
# adapted from https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
