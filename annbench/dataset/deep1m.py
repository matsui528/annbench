from .base import BaseDataset
from ..util import ivecs_read, fvecs_read

from urllib.request import urlretrieve
import tarfile

import subprocess


class Deep1m(BaseDataset):
    def __init__(self, path):
        super().__init__(path=path)

    def download(self):
        # https://github.com/matsui528/deep1b_gt#bonus-deep1m

        if self.path.exists():
            return

        self.path.mkdir(parents=True)

        root = str(self.path.resolve())
        subprocess.run(f"git clone https://github.com/matsui528/deep1b_gt.git {root}", shell=True)
        subprocess.run(f"python {root}/download_deep1b.py --root {root}/deep1b --base_n 1 --learn_n 1 --ops query base learn", shell=True)

        

        # tar_path = self.path / "sift.tar.gz"
        # if not tar_path.exists():
        #     urlretrieve("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", tar_path)
        # with tarfile.open(tar_path, 'r:gz') as f:
        #     f.extractall(path=self.path)

    def vecs_train(self):
        vec_path = self.path / "sift/sift_learn.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def vecs_base(self):
        vec_path = self.path / "sift/sift_base.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def vecs_query(self):
        vec_path = self.path / "sift/sift_query.fvecs"
        assert vec_path.exists()
        return fvecs_read(fname=str(vec_path))

    def groundtruth(self):
        vec_path = self.path / "sift/sift_groundtruth.ivecs"
        assert vec_path.exists()
        return ivecs_read(fname=str(vec_path))
