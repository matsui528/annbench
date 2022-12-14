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
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, path=self.path)

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
