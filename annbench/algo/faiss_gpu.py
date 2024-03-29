from .base import BaseANN
import faiss


class LinearGpuANN(BaseANN):
    def __init__(self):
        self.index = None

    def set_index_param(self, param):
        pass

    def has_train(self):
        return False

    def add(self, vecs):
        cpu_index = faiss.IndexFlatL2(vecs.shape[1])
        # Use all GPUs
        self.index = faiss.index_cpu_to_all_gpus(cpu_index)
        self.index.add(vecs)

    def query(self, vecs, topk, param):
        # faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        _, ids = self.index.search(x=vecs, k=topk)
        return ids

    def write(self, path):
        # faiss.write_index(self.index, path)
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), path)

    def read(self, path, D=None):
        self.index = faiss.index_cpu_to_all_gpus(faiss.read_index(path))

    def stringify_index_param(self, param):
        return "index.bin"




class IvfpqGpuANN(BaseANN):
    def __init__(self):
        self.M, self.nlist, self.index = None, None, None

    def set_index_param(self, param):
        self.M, self.nlist = param["M"], param["nlist"]

    def has_train(self):
        return True

    def train(self, vecs):
        D = vecs.shape[1]
        quantizer = faiss.IndexFlatL2(D)
        cpu_index = faiss.IndexIVFPQ(quantizer, D, self.nlist, self.M, 8)
        self.index = faiss.index_cpu_to_all_gpus(cpu_index)
        self.index.train(vecs)


    def add(self, vecs):
        self.index.add(vecs)


    def query(self, vecs, topk, param):
        #self.index.nprobe = param["nprobe"]
        #faiss.omp_set_num_threads(1)  # Make sure this is on a single thread mode
        for n_gpu in range(self.index.count()):
            faiss.downcast_index(self.index.at(n_gpu)).nprobe = param["nprobe"]
        _, ids = self.index.search(x=vecs, k=topk)
        return ids

    def write(self, path):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), path)

    def read(self, path, D=None):
        self.M, self.nlist = None, None
        cpu_index = faiss.read_index(path)
        self.M, self.nlist = cpu_index.pq.M, cpu_index.nlist
        self.index = faiss.index_cpu_to_all_gpus(cpu_index)


    def stringify_index_param(self, param):
        return f"M{param['M']}_nlist{param['nlist']}.bin"


