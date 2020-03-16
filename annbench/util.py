import numpy as np
import time


def stringify_dict(d):
    """
    d = {"a", 123, "b", "xyz", "c": "hij"}
    stringify_dict(d)
    -> "a=1, b=xyz, c=hij"
    """
    if len(d) == 0:
        return ""
    assert isinstance(d, dict)
    s = ""
    for k, v in d.items():
        s += str(k) + "=" + str(v) + ", "

    return s[:-2]  # delete the last ", "

# The following IO/eval functions are from faiss
# https://github.com/facebookresearch/faiss/blob/master/benchs/datasets.py

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def recall_at_r(I, gt, r):
    """
    Compute Recall@r over the all queries.

    Args:
        I (np.ndarray): Retrieval result, with shape(#queries, ANY), integer.
                        The index of the database item
        gt (np.ndarray): Groundtruth. np.array with shape(#queries, ANY). Integer.
                         Only gt[:, 0] is used
        r (int): Top-r

    Returns:
        The average recall@r over all queries
    """
    assert r <= I.shape[1]
    assert len(I) == len(gt)
    n_ok = (I[:, :r] == gt[:, :1]).sum()
    return n_ok / float(I.shape[0])


def evaluate(algo, vecs_query, gt, topk, r, param_query):
    """
    Run the algorithm instance for the given queries.

    Args:
        algo (BaseANN): Algorithm class which inherits BaseANN
        vecs_query (np.ndarray): Query vectors. shape=(Nq, D), dtype=np.float32
        gt (np.ndarray): Groundtruth. shape=(Nq, ANY). dtype=np.int32
        topk (int): The number of items to be returned
        r (int): r of Recall@r. Usually this equals to topk.
        param_query (ditc): A dictionary containing query parameter.

    Returns:
        duration: runtime per query (second)
        recall: recall_at_r
    """
    assert vecs_query.ndim == 2
    assert vecs_query.dtype == np.float32
    nq = vecs_query.shape[0]
    assert nq == len(gt)
    assert r <= topk

    t0 = time.time()
    ids = algo.query(vecs=vecs_query, topk=topk, param=param_query)
    t1 = time.time()

    # Each row of ids may not be strictly topk. If len(row) < topk, fill the rest by -1
    for n, row in enumerate(ids):
        if len(row) < topk:
            # Suppose row is a list (not np.array)
            ids[n] = row + [-1] * (topk - len(row))
    ids = np.array(ids)

    assert ids.shape == (nq, topk)

    return (t1 - t0) / nq, recall_at_r(I=ids, gt=gt, r=r)
