################################################################################
###
### This script uses a BertMatcher model and predicts a list of matches
###
################################################################################

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from dataset import BertMatcherDataset

def predict(matches, model, batch_size):
    matches = BertMatcherDataset(matches)
    model = torch.load(model)

    dataloader = DataLoader(matches, batch_size, sampler=SequentialSampler(matches))

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

def matcher(args):
    y_preds, y_probs = predict(args.matches, args.model, args.batch_size)

    # write to output
    block_matches = pd.read_csv(args.block_matches)
    output = []
    for i, candidate in block_matches.iterrows():
        output.append({
            "id1": candidate['id1'],
            "id2": candidate['id2'],
            "match": y_preds[i],
            "prob": y_probs[i]
        })
    output = pd.DataFrame(output)
    output.to_csv(args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--matches')
    parser.add_argument('--block-matches')
    parser.add_argument('--batch-size')
    parser.add_argument('--output')

    args = parser.parse_args()

    matcher(args)

