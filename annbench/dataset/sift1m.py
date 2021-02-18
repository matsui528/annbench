from .base import BaseDataset
from ..util import ivecs_read, fvecs_read

from urllib.request import urlretrieve
import tarfile


class Sift1m(BaseDataset):
    def __init__(self, path):
        super().__init__(path=path)

    def download(self):
        self.path.mkdir(exist_ok=True, parents=True)
        tar_path = self.path / "sift.tar.gz"
        if not tar_path.exists():
            urlretrieve("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", tar_path)
        with tarfile.open(tar_path, 'r:gz') as f:
            f.extractall(path=self.path)

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
