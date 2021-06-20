import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel

from utils.index import make_index

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def feature_extraction(model, tokenizer, sentences, device, batch_size=1000):
    def get_embeddings(sentences):
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    embedding = np.empty((0,768), int)
    for names in tqdm(batch(sentences, 1000), total = len(sentences.name) // batch_size):
        embedding = np.append(embedding, get_embeddings(names.tolist()).cpu().detach().numpy(), axis=0)
    
    return embedding
        
def make_features_index(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    products = pd.read_csv(args.products)
    feature_embedding= feature_extraction(model, tokenizer, products.name)

    index, ids_mapping = make_index(feature_embedding, products.id.values)
    index.save(args.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--products')
    parser.add_argument('-m', '--model')
    parser.add_argument('-s', '--save')
    args = parser.parse_args()

    make_features_index(args)


