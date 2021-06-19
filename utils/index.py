from annoy import AnnoyIndex
from tqdm import tqdm

def make_index(embedding, ids, trees=10, metric='angular'):
    index = AnnoyIndex(embedding.shape[1], metric)
    id_mapping = dict(zip(range(len(ids)), ids))

    for i, vec in tqdm(enumerate(embedding)):
        index.add_item(i, vec.tolist())
    index.build(trees)

    return index, id_mapping

def load_index(file, dimensions, metric='angular'):
    index = AnnoyIndex(dimensions, metric)
    index.load(file)
    return index
