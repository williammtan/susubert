from transformers import AutoTokenizer

import tensorflow as tf

def make_dataset(df, tokenizer):
    encodings = tokenizer(text=df.sent1.tolist(), text_pair=df.sent2.tolist(), truncation=True, padding='max_length', max_length=150, return_attention_mask=True, add_special_tokens = True)
    return tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        df.match.values.astype('float32').reshape((-1,1)) if 'match' in df.columns else None
    ))

def get_tokenizer(lm):
    return AutoTokenizer.from_pretrained(lm)

def read_matches(file):
    with open(file) as f:
        sents = []
        labels = []
        for line in f:
            if line == '\n':
                continue
            data = line.split('\t')
            assert len(data) <= 3, "Matches has more than 3 columns"

            sent1, sent2 = data[0], data[1]
            sents.append([sent1, sent2])

            if len(data) == 3:
                labels.append(data[2])

        return sents, labels
