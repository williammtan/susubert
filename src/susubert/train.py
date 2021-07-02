import os
from transformers import AutoTokenizer
from tensorflow import keras
import tensorflow as tf

import argparse
from transformers import TFAutoModelForSequenceClassification

from ..utils.utils import read_matches

def make_match_dataset(dataset_path, tokenizer):
    sents, labels = read_matches(dataset_path)
    sent1, sent2 = list(zip(*sents))
    input_encodings = tokenizer(text=sent1, text_pair=sent2, truncation=True, padding=True)
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(input_encodings),
        [int(l) for l in labels]
    ))
    return dataset

def create_bert_model(lm, lr):
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(lm)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy']) # can also use any keras loss fn

    return model, tokenizer

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
    model, tokenizer = create_bert_model(hp.lm, hp.lr)

    train_dataset = make_match_dataset(hp.train_set, tokenizer)
    val_dataset = make_match_dataset(hp.val_set, tokenizer)
    test_dataset = make_match_dataset(hp.test_set, tokenizer)


    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=hp.logdir)
    model.fit(
        train_dataset.shuffle(len(train_dataset)).batch(hp.batch_size), epochs=hp.n_epochs, 
        batch_size=hp.batch_size, validation_data=val_dataset.shuffle(len(val_dataset)).batch(hp.batch_size),
        callbacks=[tensorboard_callback]
    )

    print(f"<============= Test evaluation =============>")
    model.evaluate(test_dataset.batch(hp.batch_size), return_dict=True, batch_size=hp.batch_size)

    model.save_pretrained(hp.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name')
    parser.add_argument('--train-set')
    parser.add_argument('--val-set')
    parser.add_argument('--test-set')
    parser.add_argument('--lm', default='indobenchmark/indobert-base-p1')
    parser.add_argument('--n-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--save')
    parser.add_argument('--logdir', default='log/')

    args = parser.parse_args()
    train(hp=args)
