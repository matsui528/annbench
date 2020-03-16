from .base import BaseANN
import annoy


class AnnoyANN(BaseANN):
    def __init__(self):
        self.n_trees, self.index = None, None

    def __str__(self):
        return "Annoy(n_trees={}, index={})".format(self.n_trees, self.index)

    def set_index_param(self, param):
        self.n_trees = param["n_trees"]

    def has_train(self):
        return False

    def add(self, vecs):
        self.index = annoy.AnnoyIndex(f=vecs.shape[1], metric="euclidean")
        for n, vec in enumerate(vecs):
            self.index.add_item(n, vec.tolist())
        self.index.build(self.n_trees)

    def query(self, vecs, topk, param):
        return [self.index.get_nns_by_vector(vector=vec.tolist(), n=topk, search_k=param["search_k"]) for vec in vecs]

    def write(self, path):
        self.index.save(path)

    def read(self, path, D):
        self.index = annoy.AnnoyIndex(f=D, metric="euclidean")
        self.index.load(path, prefault=True)
        self.n_trees = self.index.get_n_trees()

    def stringify_index_param(self, param):
        return "ntrees{}.bin".format(param["n_trees"])

