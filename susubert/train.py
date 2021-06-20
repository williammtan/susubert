import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from susubert.model import BERTMatcher
from susubert.dataset import BertMatcherDataset

def unzip_dataloader(inputs):
    return list(zip(*inputs))

def train_epoch(model, train_set, optimizer, scheduler, batch_size):
    dataloader = DataLoader(train_set, batch_size, sampler=RandomSampler(train_set))
    criterion = nn.CrossEntropyLoss()

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, _ = batch

        logits, y_pred = model(x)
        loss = criterion(logits, y_pred) # get loss from criterion

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()


def val_epoch(model, val_set, batch_size):
    dataloader = DataLoader(val_set, batch_size, sampler=SequentialSampler(val_set))

    y_preds = np.empty(0)
    y_trues = np.empty(0)

    model.train()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, y_true = batch

        logits, y_pred, = model(x)

        y_preds = np.append(y_preds, y_pred.cpu().numpy())
        y_trues = np.append(y_trues, y_true.cpu().numpy())
    
    return y_trues, y_preds


def train(hp):
    """
    hp: Hyperparameters
        - task_name: task/run name
        - train_set, val_set, test_set: str to dataset (schema: name1, name2, match)
        - lm: language model (default: indobenchmark/indobert-base-p1)
        - n_epochs: number of epochs
        - batch_size
        - lr: learning rate
        - save: boolean to save model
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERTMatcher(hp.lm, device)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(hp.lm)

    train_set = BertMatcherDataset(hp.train_set, lm=hp.lm)
    val_set = BertMatcherDataset(hp.val_set, lm=hp.lm)
    test_set = BertMatcherDataset(hp.test_set, lm=hp.lm)

    optimizer = AdamW(model.parameters(),lr=hp.lr)
    num_steps = len(train_set) // hp.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0, num_training_steps=num_steps * hp.n_epochs)

    for i in range(hp.n_epochs):
        train_epoch(model, train_set, optimizer, scheduler, hp.batch_size)
        y_true, y_pred = val_epoch(model, val_set, hp.batch_size)
        print(f"<============= Validation Results: epoch {i+1} =============>")
        print(classification_report(y_true, y_pred))
    
    y_true, y_pred = val_epoch(test_set, val_set, hp.batch_size)
    print(f"<============= Test Results: =============>")
    print(classification_report(y_true, y_pred))

    torch.save(model.state_dict(), args.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name')
    parser.add_argument('--train-set')
    parser.add_argument('--val-set')
    parser.add_argument('--test-set')
    parser.add_argument('--lm', default='indobenchmark/indobert-base-p1')
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--save')

    args = parser.parse_args()
    train(hp=args)
