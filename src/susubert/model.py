from tensorflow import keras
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def create_bert_model(lm, lr):
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(lm)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy']) # can also use any keras loss fn

    return model, tokenizer