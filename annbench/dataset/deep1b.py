from .base import BaseDataset
from ..util import ivecs_read, fvecs_read

from urllib.request import urlretrieve
import zipfile
import subprocess


class Deep1b(BaseDataset):
    def __init__(self, path):
        super().__init__(path=path)

    # [todo] write download code

    # def download(self):
    #     # https://github.com/matsui528/deep1b_gt#bonus-deep1m

    #     root = str(self.path.resolve())

    #     # Clone deep1b_gt repository
    #     if not self.path.exists():
    #         self.path.mkdir(parents=True)
    #         subprocess.run(f"git clone https://github.com/matsui528/deep1b_gt.git {root}", shell=True)

    #     if (self.path / "deep1b/deep1B_queries.fvecs").exists() and \
    #        (self.path / "deep1b/deep1M_base.fvecs").exists() and \
    #        (self.path / "deep1b/deep1M_learn.fvecs").exists():
    #        pass
    #     else:
    #         # Download base_00, learn_00, and query on {root}/deep1b. This may take some hours. I recommend preparing 25GB of the disk space.
    #         subprocess.run(f"python {root}/download_deep1b.py --root {root}/deep1b --base_n 1 --learn_n 1 --ops query base learn", shell=True)

    #         # Select top 1M vectors from base_00 and save it on deep1M_base.fvecs
    #         subprocess.run(f"python {root}/pickup_vecs.py --src {root}/deep1b/base/base_00 --dst {root}/deep1b/deep1M_base.fvecs --topk 1000000", shell=True)

    #         # Select top 100K vectors from learn_00 and save it on deep1M_learn.fvecs
    #         subprocess.run(f"python {root}/pickup_vecs.py --src {root}/deep1b/learn/learn_00 --dst {root}/deep1b/deep1M_learn.fvecs --topk 100000", shell=True)

    #     # Download pre-computed gt
    #     gt_path = self.path / "gt.zip"
    #     if not gt_path.exists():
    #         urlretrieve("https://github.com/matsui528/deep1b_gt/releases/download/v0.1.0/gt.zip", gt_path)

    #     with zipfile.ZipFile(gt_path, 'r') as f:
    #         f.extractall(self.path)
             
    def vecs_train(self):
        import struct
        import numpy as np
        vec_path = self.path / "deep1b/learn.fvecs"
        assert vec_path.exists()
        # return fvecs_read(fname=str(vec_path))[:10 * 1000 * 1000]  # Use top 10M vectors


        # Read topk vector one by one
        vecs = []
        N = 1000 * 1000
        with vec_path.open("rb") as f:
            while 1:
                # The first 4 byte is for the dimensionality
                dim_bin = f.read(4)
                if dim_bin == b'':
                    break

                # The next 4 * dim byte is for a vector
                dim, = struct.unpack('i', dim_bin)
                vec = struct.unpack('f' * dim, f.read(4 * dim))
                vecs.append(vec)
                
                if len(vecs) == N:
                    return np.array(vecs).astype(np.float32)

    def vecs_base(self):
        import struct

        vec_path = self.path / "deep1b/base.fvecs"
        assert vec_path.exists()

        # [todo] this is slow. Shoud read batches directly?

        # Read topk vector one by one
        with vec_path.open("rb") as f:
            while 1:
                # The first 4 byte is for the dimensionality
                dim_bin = f.read(4)
                if dim_bin == b'':
                    break

                # The next 4 * dim byte is for a vector
                dim, = struct.unpack('i', dim_bin)
                vec = struct.unpack('f' * dim, f.read(4 * dim))
                
                yield vec

    def vecs_query(self):
        vec_path = self.path / "deep1b/deep1B_queries.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def groundtruth(self):
        vec_path = self.path / "deep1b/deep1B_groundtruth.ivecs"
        assert vec_path.exists()
        return ivecs_read(fname=str(vec_path))


