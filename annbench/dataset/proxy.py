from .siftsmall import Siftsmall
from .sift1m import Sift1m
from .deep1m import Deep1m
from .deep1b import Deep1b

def instantiate_dataset(name, path):
    """
    Instantiate a dataset class
    Args:
        name: the name of the target dataset
        path: the path to the directory to store the dataset

    Returns:
        an instance of the specified dataset class
    """
    if name == "siftsmall":
        return Siftsmall(path=path)
    elif name == "sift1m":
        return Sift1m(path=path)
    elif name == "deep1m":
        return Deep1m(path=path)
    elif name == "deep1b":
        return Deep1b(path=path)
    else:
        return None



