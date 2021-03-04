from .base import BaseANN
import scann
from pathlib import Path

# Some hypter-parameters are from https://github.com/facebookresearch/faiss/blob/master/benchs/bench_all_ivf/cmp_with_scann.py
# This SCANN does not include re-order process

class ScannANN(BaseANN):
    def __init__(self):
        self.num_leaves, self.reorder, self.index = None, None, None

    def set_index_param(self, param):
        self.num_leaves = param["num_leaves"]  # ~ sqrt(N)
        self.reorder = param["reorder"]

    def has_train(self):
        return False

    def add(self, vecs):
        nt = min(vecs.shape[0], 250000)
        thr = 0

        sb = scann.scann_ops_pybind.builder(db=vecs, num_neighbors=10, distance_measure="squared_l2")
        sb.tree(num_leaves=self.num_leaves, num_leaves_to_search=100, training_sample_size=nt)
        sb.score_ah(dimensions_per_block=2, anisotropic_quantization_threshold=thr)

        # Re-compute based on the actual vectors?
        if 0 < self.reorder:
            sb.reorder(self.reorder)

        self.index = sb.build()


    def query(self, vecs, topk, param):
        ids, _ = self.index.search_batched(vecs, leaves_to_search=param["nprobe"], final_num_neighbors=topk)
        # Note: There exists a function .search_batched_parallel() as well.
        return ids

    def write(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.index.serialize(str(p))

    def read(self, path, D):
        # self.num_leaves is not set after reading
        self.index = scann.scann_ops_pybind.load_searcher(path)

    def stringify_index_param(self, param):
        return f"nleaves{param['num_leaves']}_reorder{param['reorder']}"

