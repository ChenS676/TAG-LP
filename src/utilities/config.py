import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'cora'
    # Cuda device number, used for machine with multiple gpus
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Number of runs with random init
    cfg.runs = 4
    
    cfg.local_rank = 0
    cfg.fp16 = False
    
    cfg.data = CN()
    cfg.data.dir = 'umls'

    cfg.task_name = 'kg'
    cfg.cache_dir = ''
    cfg.output_dir = './output_umls/'
    # ------------------------------------------------------------------------ #
    # Train options check demo
    # ------------------------------------------------------------------------ #
    # cfg.do_train = True
    # cfg.do_eval = True
    # cfg.do_predict = True
    # cfg.train_batch_size = 128
    # cfg.gradient_accumulation_steps = 1
    # # cfg.num_train_epochs = 3.0    
    # cfg.eval_batch_size = 8 
    
    
    cfg.gnn = CN()
    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    # GNN model name
    cfg.gnn.model.name = 'GCN'
    # Number of gnn layers
    cfg.gnn.model.num_layers = 4
    # Hidden size of the model
    cfg.gnn.model.hidden_dim = 128

    # ------------------------------------------------------------------------ #
    # GNN Training options
    # ------------------------------------------------------------------------ #
    cfg.gnn.train = CN()
    # Use PyG or DGL
    cfg.gnn.train.use_dgl = False
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.gnn.train.weight_decay = 0.0
    # Maximal number of epochs
    cfg.gnn.train.epochs = 200
    # Node feature type, options: ogb, TA, P, E
    cfg.gnn.train.feature_type = 'TA_P_E'
    # Number of epochs with no improvement after which training will be stopped
    cfg.gnn.train.early_stop = 50
    # Base learning rate
    cfg.gnn.train.lr = 0.01
    # L2 regularization, weight decay
    cfg.gnn.train.wd = 0.0
    # Dropout rate
    cfg.gnn.train.dropout = 0.0

    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm = CN()
    cfg.lm.model = CN()
    # LM model name
    cfg.lm.model.name = 'bert-base-uncased'

    # Set this flag if you are using an uncased model.
    cfg.lm.do_lower_case = False 
    cfg.lm.model.feat_shrink = ""

    cfg.lm.learning_rate = 5e-5
    cfg.lm.warmup_proportion = 0.1
    
    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    #  Number of samples computed once per batch per device
    # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.grad_acc_steps = 1
    # Base learning rate
    
    # Maximal number of epochs
    cfg.lm.train.epochs = 2
    # The number of warmup steps
    cfg.lm.train.warmup_epochs = 0.6
    # Number of update steps between two evaluations
    cfg.lm.train.eval_patience = 50000
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.lm.train.weight_decay = 0.0
    # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    cfg.lm.train.dropout = 0.3
    # The dropout ratio for the attention probabilities
    cfg.lm.train.att_dropout = 0.1
    # The dropout ratio for the classifier
    cfg.lm.train.cla_dropout = 0.4
    # Whether or not to use the gpt responses (i.e., explanation and prediction) as text input
    # If not, use the original text attributes (i.e., title and abstract)
    cfg.lm.train.use_gpt = False

    
    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, cfg_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], ncfg=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(cfg_str, str):
        # parse from a string
        cfg = parser.parse_cfg(cfg_str.split())
    else:
        # parse from command line
        cfg = parser.parse_cfg()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(cfg.config):
        cfg.merge_from_file(cfg.config)

    # Update from command line
    cfg.merge_from_list(cfg.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())