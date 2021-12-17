import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

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

    match_dataset = make_dataset(matches, tokenizer)
    logits = model.predict(match_dataset.batch(args.batch_size), batch_size=args.batch_size, verbose=1).logits
    logits = softmax(logits).numpy()

    probabilities = logits[:, 1]
    predictions = (probabilities > args.threshold).astype(int)

    match_results = pd.DataFrame({'id1': matches.id1.values, 'id2': matches.id2.values, 'match': predictions, 'prob': probabilities})
    return match_results


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
