################################################################################
###
### This script trains a S-BERT model
###
################################################################################

from sentence_transformers import SentenceTransformer, models, losses
from dataset import SBertMatcherDataset
from torch.utils.data import DataLoader
import argparse


def train_blocker(hp):
    word_embedding_model = models.Transformer(hp.lm)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    dataset = SBertMatcherDataset(hp.data, lm=hp.lm)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
    loss_func = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(dataloader, loss_func)], epochs=1, warmup_steps=100)

    model.save(hp.save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm')
    parser.add_argument('--data')
    parser.add_argument('--save')
    args = parser.parse_args()

    train_blocker(args)