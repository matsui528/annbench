from pathlib import Path

class BaseDataset(object):
    def __init__(self, path):
        self.path = Path(path)

    def download(self):
        pass

    def vecs_train(self):
        # Return a vector (np.array with NxD) or a generator
        pass

    def vecs_base(self):
        # Return a vector (np.array with NxD) or a generator
        pass

    def vecs_query(self):
        # Return a vector (np.array with NxD) or a generator
        pass

    def groundtruth(self):
        # Return a vector (np.array with NxD) or a generator
        pass

    def D(self):
        vecs = self.vecs_train()
        return vecs.shape[1]
