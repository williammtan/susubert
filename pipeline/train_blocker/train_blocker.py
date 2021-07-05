import pandas as pd
import numpy as np
import argparse

from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from dataset import SBertMatcherDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

VAL_SPLIT=0.1 # TODO: make this a parameter

def create_sbert_model(lm, max_length=64):
    word_embedding_model = models.Transformer(lm, max_seq_length=max_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

def train_blocker(lm, train_matches, val_matches, args):
    model = create_sbert_model(lm)

    dataset = SBertMatcherDataset(np.array([train_matches.sent1, train_matches.sent2]).T, labels=train_matches.match.values, lm=lm)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    loss_func = losses.CosineSimilarityLoss(model)

    val_evaluator = BinaryClassificationEvaluator(sentences1=val_matches.sent1.tolist(), sentences2=val_matches.sent2.tolist(), labels=val_matches.match.tolist())

    model.fit(train_objectives=[(dataloader, loss_func)], epochs=args.n_epochs, evaluator=val_evaluator)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches')
    parser.add_argument('--lm')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--save-model')
    args = parser.parse_args()

    matches = pd.read_csv(args.matches)
    train_matches, val_matches = train_test_split(matches, test_size=VAL_SPLIT)

    model = train_blocker(args.lm, train_matches, val_matches, args)
    model.save(args.save_model)
