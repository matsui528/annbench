from .base import BaseANN
import pickle
import pynndescent


class PynndescentANN(BaseANN):
    def __init__(self):
        self.ef_construction, self.M, self.index = None, None, None

    def set_index_param(self, param):
        self.n_neighbors = param["n_neighbors"]
        self.diversify_prob = param["diversify_prob"]
        self.pruning_degree_multiplier = param["pruning_degree_multiplier"]

    def has_train(self):
        return False

    def add(self, vecs):
        self.index = pynndescent.NNDescent(
            vecs,
            n_neighbors=self.n_neighbors,
            diversify_prob=self.diversify_prob,
            pruning_degree_multiplier=self.pruning_degree_multiplier,
            low_memory=True,
            compressed=True,
            verbose=True,
        )
        self.index.prepare()

    def query(self, vecs, topk, param):
        labels, _ = self.index.query(vecs, k=topk, epsilon=param["epsilon"])
        return labels

    def write(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.index, f)

    def read(self, path, D):
        with open(path, "rb") as f:
            self.index = pickle.load(f)

    def stringify_index_param(self, param):
        return f"n{param['n_neighbors']}_dp{param['diversify_prob']}_pdm{param['pruning_degree_multiplier']}.bin"
