################################################################################
###
### This script trains a S-BERT model
###
################################################################################

from sentence_transformers import SentenceTransformer, models, losses
from blocking.dataset import SBertMatcherDataset
from torch.utils.data import DataLoader
import numpy as np
import argparse

def create_sbert_model(lm, max_length=64):
    word_embedding_model = models.Transformer(lm, max_seq_length=max_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

def train_blocker(lm, matches, max_length=64):
    model = create_sbert_model(lm, max_length)

    matches = matches.sample(frac=1) # shuffle
    dataset = SBertMatcherDataset(np.array([matches.sent1, matches.sent2]).T, labels=matches.match, lm=lm)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
    loss_func = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(dataloader, loss_func)], epochs=2, warmup_steps=100)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm')
    parser.add_argument('--data')
    parser.add_argument('--save')
    args = parser.parse_args()

    train_blocker(args)
