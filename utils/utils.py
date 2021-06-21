from transformers import AutoTokenizer

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
