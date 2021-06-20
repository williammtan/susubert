import torch
from torch.utils.data import Dataset
from sentence_transformers import InputExample
from utils.utils import get_tokenizer
import pandas as pd
import os

class BertMatcherDataset(Dataset):
    def __init__(self, dataset, labels=None, lm=None, max_len=150):
        """
        Inputs:
            dataset: str (to file) or list
            labels: list of labels
        """
        if type(dataset) == str:
            self.sents, self.labels = self.read_data(dataset)

        elif type(dataset) == list:
            self.sents = dataset
            self.labels = labels
        
        assert len(self.sents) == len(self.labels), "Length of sents and labels don't match"

        self.lm = lm
        self.max_len = max_len
        self.tokenizer = get_tokenizer(self.lm)
        self.tokenizer.add_tokens(['COL', 'VAL'], special_tokens=True)
    
    def read_data(self, fname):
        file_extension = os.path.splitext(fname)[1]
        sents = []
        labels = []

        if file_extension == '.txt':
            for row in open(fname):
                cols = row.split('\t')
                if len(cols) == 3:
                    sents.append(cols[:2])
                    labels.append(int(cols[-1].replace('\n','')))
        elif file_extension == '/csv':
            pd.read_csv()

        return sents, labels
    
    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sents = self.sents[idx]

        x = self.tokenizer.encode(
            text=sents[0], text_pair=sents[1],
            add_special_tokens=True, 
            truncation="longest_first", 
            max_length=self.max_len, 
            return_tensors='pt', 
            pad_to_max_length=True,
        )

        if self.labels is not None:
            y = self.labels[idx]
        else:
            y = None

        return x, y

class SBertMatcherDataset(BertMatcherDataset):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = InputExample(texts=self.sents[idx], label=float(self.labels[idx]))

        return x