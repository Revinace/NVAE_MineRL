import numpy as np
import faiss


class FaissKNeighbors:
    def __init__(self):
        self.index = None

    def fit(self, X):
        nlist = 100
        m = 8
        d = X.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        self.index.train(X.astype(np.float32))
        self.index.add(X.astype(np.float32))

    def predict(self, X, k):
        distances, indices = self.index.search(X.astype(np.float32), k=k)
        return distances, indices