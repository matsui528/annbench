import psutil

class BaseANN(object):

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in bytes), or None if this information is not available."""
        # This function is from:
        # https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/base.py
        return psutil.Process().memory_info().rss

    def set_index_param(self, param):
        pass

    def has_train(self):
        pass

    def train(self, vecs):
        pass

    def add(self, vecs):
        pass

    def query(self, vecs, topk, param):
        pass

    def write(self, path):
        pass

    def read(self, path, D):
        # Some algorithms require D to read an index
        pass

    def stringify_index_param(self, param):
        # Convert param_index to an identifier.
        # The identifier will be used to name the index when writing it on the disk.
        # e.g., if param={"M": 10, "K": 8}, return "M10_K8.bin".
        pass

