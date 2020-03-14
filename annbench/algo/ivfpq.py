from .base import BaseANN

# Import faiss only if it has been installed
import importlib.util
spec = importlib.util.find_spec("faiss")
if spec is None:
    pass  # If faiss hasn't been installed. Just skip.
else:
    import faiss


class IvfpqANN(BaseANN):
    def __init__(self):
        self.M, self.nlist, self.index = None, None, None

    def __str__(self):
        return "Ivfpq(M={}, nlist={}, index={})".format(
            self.M, self.nlist, self.index
        )

    def set_index_param(self, param):
        self.M, self.nlist = param["M"], param["nlist"]

    def has_train(self):
        return True

    def train(self, vecs):
        D = vecs.shape[1]
        quantizer = faiss.IndexFlatL2(D)
        self.index = faiss.IndexIVFPQ(quantizer, D, self.nlist, self.M, 8)
        self.index.train(vecs)

    def add(self, vecs):
        self.index.add(vecs)

    def query(self, vecs, topk, param):
        self.index.nprobe = param["nprobe"]
        faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        _, ids = self.index.search(x=vecs, k=topk)
        return ids

    def write(self, path):
        faiss.write_index(self.index, path)

    def read(self, path, D=None):
        self.M, self.nlist = None, None
        self.index = faiss.read_index(path)
        self.M, self.nlist = self.index.pq.M, self.index.nlist




