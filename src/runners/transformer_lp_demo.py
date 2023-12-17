# this script is used to run the transformer model on the demo data 
# with dataloader implemented in hf_loader.py and example.py

from src.LMs.lm_trainer import LMTrainer
from src.utilities.config import cfg, update_cfg
import pandas as pd


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    for seed in seeds:
        cfg.seed = seed
        trainer = LMTrainer(cfg)
        trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)