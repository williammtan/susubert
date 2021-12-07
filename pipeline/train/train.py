# convert {'id1': 1, 'id2': 2, 'match': 0} -> 
# [CLS] [COL] title [VAL] susu indomilk 200ml [COL] price [VAL] 13000
# [SEP] [COL] title [VAL] susu beruang 200ml [COL] price [VAL] 15000 [SEP]
# , 0

import pandas as pd
import argparse
import torch
from pathlib import Path

from tensorflow import keras
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from sklearn.model_selection import train_test_split

VAL_SPLIT=0.1 # TODO: make this a parameter
LOG_DIR = 'logs/'

def create_bert_model(lm, lr, model_path):
    if model_path != '': 
        print('using pretrained model path')
        try:
            model = TFAutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        except Exception:
            model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)
    else:
        print('not using pretrained model')
        model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)

    tokenizer = AutoTokenizer.from_pretrained(lm)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']) # can also use any keras loss fn

    return model, tokenizer

def make_dataset(df, tokenizer):
    encodings = tokenizer(text=df.sent1.tolist(), text_pair=df.sent2.tolist(), truncation=True, padding='max_length', max_length=150, return_attention_mask=True, add_special_tokens = True)
    return tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        df.match.values.astype('float32').reshape((-1,1)) if 'match' in df.columns else None
    ))

def train(train_matches, val_matches, hp):
    model, tokenizer = create_bert_model(hp.lm, hp.lr, model_path=hp.model)
    print(model.summary())

    train_dataset = make_dataset(train_matches, tokenizer)
    val_dataset = make_dataset(val_matches, tokenizer)

    model.fit(
        train_dataset.shuffle(len(train_dataset)).batch(hp.batch_size), epochs=hp.n_epochs, 
        batch_size=hp.batch_size, validation_data=val_dataset.shuffle(len(val_dataset)).batch(hp.batch_size),
    )

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches')
    parser.add_argument('--lm')
    parser.add_argument('--model', required=False, default='')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--save-model')
    args = parser.parse_args()

    matches = pd.read_csv(args.matches)
    train_matches, val_matches = train_test_split(matches, test_size=VAL_SPLIT)

    model = train(train_matches, val_matches, args)
    model.save_pretrained(args.save_model)
