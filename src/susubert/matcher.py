################################################################################
###
### This script uses a BertMatcher model and predicts a list of matches
###
################################################################################

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from tensorflow.nn import softmax


def matcher(lm, model, matches, block_matches, batch_size, threshold=0.5):
    match_results = []
    def predict_batch(batch, model, tokenizer):
        """Predict batch and write to file"""
        input_encodings = tokenizer(text=batch.sent1.tolist(), text_pair=batch.sent2.tolist(), truncation=True, padding=True)
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(input_encodings)
        ))
        logits = model.predict(dataset.batch(32), batch_size=32).logits
        logits = softmax(logits).numpy() # [[0.3, 0.7], [0.4, 0.6], [0.9, 0.1]]
        # y_preds = np.argmax(logits, axis=1) # [1,1,0]

        for idx in range(len(batch)):
            match = 1 if logits[idx][1] > threshold else 0
            match_results.append({
                "id1": batch.iloc[idx].id1,
                "id2": batch.iloc[idx].id2,
                "match": match,
                "prob": logits[idx][1]
            })

    model = TFAutoModelForSequenceClassification.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(lm)

    candidates = pd.merge(block_matches, matches, left_index=True, right_index=True)
    for i in tqdm(range(0, len(candidates), batch_size)):
        batch_candidates = candidates[i:i+batch_size]
        predict_batch(batch_candidates, model, tokenizer)
    
    return pd.DataFrame(match_results)

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

