import numpy as np
import pandas as pd

from sixbirds_cosmo.lss.theory.reference import ReferenceVectorBackend
from sixbirds_cosmo.lss.theory.stub import StubBackend


def test_stub_backend_matches_definition():
    y = np.array([1.0, 2.0, 3.0])
    cov = np.diag([4.0, 9.0, 16.0])
    backend = StubBackend(y, cov, frac_sigma=0.1)
    block_index = pd.DataFrame({"i": [0, 1, 2]})
    pred = backend.predict(np.array([]), block_index)
    sigma = np.sqrt(np.diag(cov))
    assert np.allclose(pred, y + 0.1 * sigma)


def test_reference_backend_loads_npy_and_matches_length(tmp_path):
    vec = np.array([0.1, 0.2, 0.3])
    path = tmp_path / "ref.npy"
    np.save(path, vec)
    backend = ReferenceVectorBackend(ref_path=path)
    block_index = pd.DataFrame({"i": [0, 1, 2]})
    pred = backend.predict(np.array([]), block_index)
    assert np.allclose(pred, vec)


def test_reference_backend_subsets_by_block_index_i(tmp_path):
    vec = np.arange(10, dtype=float)
    path = tmp_path / "ref.npy"
    np.save(path, vec)
    backend = ReferenceVectorBackend(ref_path=path)
    block_index = pd.DataFrame({"i": [2, 5, 9]})
    pred = backend.predict(np.array([]), block_index)
    assert np.allclose(pred, vec[[2, 5, 9]])
