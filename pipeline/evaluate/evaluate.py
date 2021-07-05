import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import json

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import classification_report

def create_bert_model(model_path, lm):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(lm)

    return model, tokenizer

def make_dataset(df, tokenizer):
    encodings = tokenizer(text=df.sent1.tolist(), text_pair=df.sent2.tolist(), truncation=True, padding='max_length', max_length=150, return_attention_mask=True, add_special_tokens = True)
    return tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        df.match.values.astype('float32').reshape((-1,1)) if 'match' in df.columns else None
    ))

def evaluate(matches, model, lm, batch_size):
    model, tokenizer = create_bert_model(args.model, args.lm)

    dataset = make_dataset(matches, tokenizer)
    y_pred = model.predict(dataset.batch(args.batch_size), batch_size=batch_size).logits

    return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches')
    parser.add_argument('--lm')
    parser.add_argument('--model')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--mlpipeline-ui-metadata-path')
    args = parser.parse_args()

    matches = pd.read_csv(args.matches)
    y_pred = evaluate(matches=matches, model=args.model, lm=args.lm, batch_size=args.batch_size)

    class_report = classification_report(matches.match.values, np.argmax(y_pred, axis=1))
    markdown = "# Evaluation results \n " + class_report

    metadata = {
        'outputs' : [
        {
            'storage': 'inline',
            'source': markdown,
            'type': 'markdown',
        }]
    }

    Path(args.ml_pipeline_ui_metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.mlpipeline_ui_metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)
