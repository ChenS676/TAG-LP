from transformers import data as hf_data
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from IPython import embed

# adopted from https://github.com/ChenS676/TAPE/blob/main/core/LMs/lm_trainer.py
class Dataset(Dataset):
    """Dataset class for kg embedding for dataset:

    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    raise NotImplementedError

