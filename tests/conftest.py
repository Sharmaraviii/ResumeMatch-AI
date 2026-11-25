"""In-memory FAISS stand-in when native FAISS is unavailable in test environment."""
import importlib.util
import sys
import types
import numpy as np

if importlib.util.find_spec('faiss') is None:
    module = types.ModuleType('faiss')
    def normalize_L2(vectors):
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    class IndexFlatIP:
        def __init__(self, dimension):
            self.dimension = dimension
        def add(self, vectors):
            self.vectors = vectors.copy()
        def search(self, queries, k):
            similarities = queries @ self.vectors.T
            indices = np.argsort(-similarities, axis=1)[:, :k]
            return np.take_along_axis(similarities, indices, axis=1), indices
    module.normalize_L2 = normalize_L2
    module.IndexFlatIP = IndexFlatIP
    sys.modules['faiss'] = module
