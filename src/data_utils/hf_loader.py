from transformers import data as hf_data
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from IPython import embed

# adopted from https://github.com/ChenS676/TAPE/blob/main/core/LMs/lm_trainer.py
import torch
from src.data_utils.dataset import Dataset

class Dataset(Dataset):
    """Dataset class for kg embedding for dataset:

    """
    def __init__(self, X, cfg):
        self.output_dir = f'output/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        data, num_classes, text = load_data(
            dataset=self.dataset_name, use_text=True, use_gpt=cfg.lm.train.use_gpt, seed=self.seed)
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)
        dataset = Dataset(X, data.y.tolist())
        self.inf_dataset = dataset

        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    raise NotImplementedError

