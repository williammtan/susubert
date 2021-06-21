################################################################################
###
### This script uses a BertMatcher model and predicts a list of matches
###
################################################################################

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import jsonlines

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModel

from susubert.dataset import BertMatcherDataset

def predict(matches, model, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    dataloader = DataLoader(matches, batch_size, sampler=SequentialSampler(matches))

    y_preds = np.empty(0)
    y_probs = np.empty(0)

    softmax = nn.Softmax(2)

    model.train()
    for i, x in enumerate(dataloader):

        logits, y_pred, = model(x)
        logits = logits.reshape(logits.shape[0], 1 , 2)
        logits = softmax(logits)

        y_preds = np.append(y_preds, y_pred.cpu().detach().numpy())
        y_probs = np.append(y_probs, logits[y_pred].cpu().detach().numpy()) # get probability of prediction
    
    return y_preds, y_probs

def matcher(args):
    def predict_batch(x, model, block_matches, writer):
        """Predict batch and write to file"""
        matches = BertMatcherDataset(x, lm=args.lm)
        y_preds, y_probs = predict(matches, model, 32)

        for idx in range(len(block_matches)):
            output = {
                "id1": block_matches[idx][0],
                "id2": block_matches[idx][1],
                "match": y_preds[idx],
                "prob": y_probs[idx]
            }
            writer.write(output)


    model = AutoModel.from_pretrained(args.lm)
    model = torch.load(args.model)

    with jsonlines.open(args.output, 'w') as writer, open(args.block_matches) as block_f, open(args.matches) as matches_f:
        blocks = []
        matches = []
        next(block_f) # skip header row
        for block, match in tqdm(zip(block_f, matches_f), total=18758):
            blocks.append(block.split(',')) # eg. [[2341351, 1351325], [12415131, 135135]]
            matches.append(match)

            if len(matches) == args.batch_size:
                predict_batch(matches, model, blocks, writer)
                blocks.clear()
                matches.clear()
        
        if len(matches) > 0:
            predict_batch(matches, model, blocks, writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--lm')
    parser.add_argument('--matches')
    parser.add_argument('--block-matches')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--output')

    args = parser.parse_args()

    matcher(args)

