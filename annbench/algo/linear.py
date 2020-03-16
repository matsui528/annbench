from .base import BaseANN

# Import faiss only if it has been installed
import importlib.util
spec = importlib.util.find_spec("faiss")
if spec is None:
    pass  # If faiss hasn't been installed. Just skip.
else:
    import faiss


class LinearANN(BaseANN):
    def __init__(self):
        self.index = None

    def __str__(self):
        return "Linear(index={})".format(self.index)

    def set_index_param(self, param):
        pass

    def has_train(self):
        return False

    def add(self, vecs):
        self.index = faiss.IndexFlatL2(vecs.shape[1])
        self.index.add(vecs)

    def query(self, vecs, topk, param):
        faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        _, ids = self.index.search(x=vecs, k=topk)
        return ids

    def write(self, path):
        faiss.write_index(self.index, path)

    def read(self, path, D=None):
        self.index = faiss.read_index(path)

    def stringify_index_param(self, param):
        return "index.bin"



