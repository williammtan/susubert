import tensorflow as tf

def make_dataset(df, tokenizer):
    encodings = tokenizer(text=df.sent1.tolist(), text_pair=df.sent2.tolist(), truncation=True, padding='max_length', max_length=150)
    return tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        df.match.values.astype('float32').reshape((-1,1))
    ))