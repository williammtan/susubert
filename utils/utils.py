from transformers import AutoTokenizer

def get_tokenizer(lm):
    return AutoTokenizer.from_pretrained(lm)
