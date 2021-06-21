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

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf


def matcher(args):
    def predict_batch(x, model, tokenizer, block_matches, writer):
        """Predict batch and write to file"""
        input_encodings = tokenizer(text=[m[0] for m in matches], text_pair=[m[1] for m in matches], truncation=True, padding=True)
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(input_encodings)
        ))
        logits = model.predict(dataset.batch(32))
        y_preds = np.argmax(logits, axis=1)
        y_probs = logits[y_preds]

        for idx in range(len(block_matches)):
            output = {
                "id1": block_matches[idx][0],
                "id2": block_matches[idx][1],
                "match": y_preds[idx],
                "prob": y_probs[idx]
            }
            writer.write(output)

    model = TFAutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.lm)

    with jsonlines.open(args.output, 'w') as writer, open(args.block_matches) as block_f, open(args.matches) as matches_f:
        blocks = []
        matches = []
        next(block_f) # skip header row
        for block, match in tqdm(zip(block_f, matches_f)):
            blocks.append(block.split(',')) # eg. [[2341351, 1351325], [12415131, 135135]]
            matches.append(match)

            if len(matches) == args.batch_size:
                predict_batch(matches, model, tokenizer, blocks, writer)
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

