import importlib.util


def check_module(module_name):
    # Return true only if the module has been installed
    spec = importlib.util.find_spec(module_name)
    if spec is None:  # The module hasn't been installed
        return False
    else:
        return True


if check_module("annoy"):
    from .annoy import AnnoyANN

if check_module("hnswlib"):
    from .hnsw import HnswANN

if check_module("faiss"):
    from .faiss_cpu import (
        LinearANN,
        IvfpqANN,
        Ivfpq4bitANN,
        PqANN,
        Pq4bitANN,
        HnswFaissANN,
    )
    from .faiss_gpu import LinearGpuANN, IvfpqGpuANN

if check_module("scann"):
    from .scann import ScannANN


def instantiate_algorithm(name):
    """
    Instantiate an algorithm class
    Args:
        name: the name of the target algorithm

    Returns:
        an instance of the specified algorithm class
    """

    if name == "annoy":
        return AnnoyANN()
    elif name == "ivfpq":
        return IvfpqANN()
    elif name == "hnsw":
        return HnswANN()
    elif name == "linear":
        return LinearANN()
    elif name == "linear_gpu":
        return LinearGpuANN()
    elif name == "ivfpq_gpu":
        return IvfpqGpuANN()
    elif name == "ivfpq4bit":
        return Ivfpq4bitANN()
    elif name == "pq":
        return PqANN()
    elif name == "pq4bit":
        return Pq4bitANN()
    elif name == "hnsw_faiss":
        return HnswFaissANN()
    elif name == "scann":
        return ScannANN()
    else:
        return None
