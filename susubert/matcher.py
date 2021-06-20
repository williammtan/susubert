################################################################################
###
### This script uses a BertMatcher model and predicts a list of matches
###
################################################################################

import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from dataset import BertMatcherDataset

def predict(hp):
    matches = BertMatcherDataset(hp.matches)
    model = torch.load(hp.model)

    dataloader = DataLoader(matches, hp.batch_size, sampler=SequentialSampler(matches))

    y_preds = np.empty(0)
    y_probs = np.empty(0)

    softmax = nn.Softmax(2)

    model.train()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x, _ = batch

        logits, y_pred, = model(x)
        logits = softmax(logits)

        y_pred = y_pred.cpu().numpy()

        y_preds = np.append(y_preds, y_pred)
        y_probs = np.append(y_probs, logits[y_pred]) # get probability of prediction
    
    return y_preds, y_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm')
    parser.add_argument('--model')
    parser.add_argument('--matches')
    parser.add_argument('--batch-size')
    parser.add_argument('--output')

    args = parser.parse_args()

    y_preds, y_probs = predict(args)

    # write to output
    output = open(args.output, 'w')
    for y_pred, y_prob in zip(y_preds, y_probs):
        output.write(str(y_pred) + '\t' + y_probs + '\n')

