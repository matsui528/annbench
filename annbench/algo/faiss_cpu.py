from .base import BaseANN
import faiss



class LinearANN(BaseANN):
    def __init__(self):
        self.index = None

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


class PqANN(BaseANN):
    def __init__(self):
        self.M, self.index = None, None

    def set_index_param(self, param):
        self.M = param["M"]

    def has_train(self):
        return True

    def train(self, vecs):
        self.index = faiss.IndexPQ(vecs.shape[1], self.M, 8)
        self.index.train(vecs)

    def add(self, vecs):
        self.index.add(vecs)

    def query(self, vecs, topk, param):
        faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        _, ids = self.index.search(x=vecs, k=topk)
        return ids

    def write(self, path):
        faiss.write_index(self.index, path)

    def read(self, path, D=None):
        self.index = faiss.read_index(path)
        self.M = self.index.pq.M

    def stringify_index_param(self, param):
        return f"M{param['M']}.bin"

class Pq4bitANN(PqANN):
    def train(self, vecs):
        self.index = faiss.IndexPQFastScan(vecs.shape[1], self.M, 4)
        self.index.train(vecs)


class IvfpqANN(BaseANN):
    def __init__(self):
        self.M, self.nlist, self.quantizer, self.index = None, None, None, None

    def set_index_param(self, param):
        self.M, self.nlist, self.quantizer = param["M"], param["nlist"], param["quantizer"]


    def has_train(self):
        return True

    def train(self, vecs):
        D = vecs.shape[1]
        
        if self.quantizer == "flat":
            quantizer = faiss.IndexFlatL2(D)
        elif self.quantizer == "hnsw":
            quantizer = faiss.IndexHNSWFlat(D, 32)
        else:
            assert 0, f"the quantizer name is strange: {self.quantizer}"

        self.index = faiss.IndexIVFPQ(quantizer, D, self.nlist, self.M, 8)
        self.index.train(vecs)

    def add(self, vecs):
        if hasattr(vecs, '__iter__'):  # if vecs is iterator such as Deeb1B
            from more_itertools import chunked
            import numpy as np

            # [todo] N is fixed for 1B. To be updated.
            batch_size = 1000 * 1000
            N = 1000 * 1000 * 1000

            for n, vecs_batch in enumerate(chunked(vecs, batch_size)):
                print(f"{n}/{int(N/batch_size)}")
                self.index.add(np.array(vecs_batch).astype(np.float32))
        else:
            self.index.add(vecs)




    def query(self, vecs, topk, param):
        self.index.nprobe = param["nprobe"]
        faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        _, ids = self.index.search(x=vecs, k=topk)
        return ids

    def write(self, path):
        faiss.write_index(self.index, path)

    def read(self, path, D=None):
        self.M, self.nlist, self.quzntizer = None, None, None
        self.index = faiss.read_index(path)
        self.M, self.nlist = self.index.pq.M, self.index.nlist
        if "flat" in path:
            self.quantizer = "flat"
        elif "hnsw" in path:
            self.quantizer = "hnsw"
        else:
            assert 0, f"the path doesn't contain a proper quantizer name: {self.path}"


    def stringify_index_param(self, param):
        return f"M{param['M']}_nlist{param['nlist']}_quantizer_{param['quantizer']}.bin"


class Ivfpq4bitANN(IvfpqANN):
    def train(self, vecs):
        D = vecs.shape[1]
        
        if self.quantizer == "flat":
            quantizer = faiss.IndexFlatL2(D)
        elif self.quantizer == "hnsw":
            quantizer = faiss.IndexHNSWFlat(D, 32)
        else:
            assert 0, f"the quantizer name is strange: {self.quantizer}"

        self.index = faiss.IndexIVFPQFastScan(quantizer, D, self.nlist, self.M, 4)
        self.index.train(vecs)