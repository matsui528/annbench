from .base import BaseANN

# Import faiss only if it has been installed
import importlib.util
spec = importlib.util.find_spec("faiss")
if spec is None:
    pass  # If faiss hasn't been installed. Just skip.
else:
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



