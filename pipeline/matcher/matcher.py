import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import math

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from tensorflow.nn import softmax # type: ignore

def create_bert_model(model_path, lm):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(lm)

    return model, tokenizer

def make_dataset(df, tokenizer):
    encodings = tokenizer(text=df.sent1.tolist(), text_pair=df.sent2.tolist(), truncation=True, padding='max_length', max_length=100, return_attention_mask=True, add_special_tokens = True)
    return tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        df.match.values.astype('float32').reshape((-1,1)) if 'match' in df.columns else None
    ))

def matcher(matches, model, args):
    model, tokenizer = create_bert_model(model, args.lm)

    match_results_full = []

    for chunk in np.array_split(matches, math.ceil(len(matches) / 50000)):
        match_dataset = make_dataset(chunk, tokenizer)
        logits = model.predict(match_dataset.batch(args.batch_size), batch_size=args.batch_size, verbose=1).logits
        logits = softmax(logits).numpy()

        probabilities = logits[:, 1]
        predictions = (probabilities > args.threshold).astype(int)

        match_results = pd.DataFrame({'id1': chunk.id1.values, 'id2': chunk.id2.values, 'match': predictions, 'prob': probabilities})
        match_results_full.append(match_results)
    match_results_full = pd.concat(match_results_full)
    return match_results_full


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches')
    parser.add_argument('--lm')
    parser.add_argument('--model')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--save-matches')
    args = parser.parse_args()

    matches = pd.read_csv(args.matches)
    if not matches.empty:
        match_results = matcher(matches, args.model, args)

        Path(args.save_matches).parent.mkdir(parents=True, exist_ok=True)
        match_results.to_csv(args.save_matches, index=False)
    else:
        Path(args.save_matches).parent.mkdir(parents=True, exist_ok=True)
        open(args.save_matches, 'w')
        print('empty match candidates, skipping')
