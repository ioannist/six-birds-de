import numpy as np

from sixbirds_cosmo.datasets.des_sn5yr import parse_triplet_cov


def test_parse_triplet_cov_one_based(tmp_path):
    content = """
1 1 4
1 2 1
2 2 9
"""
    path = tmp_path / "cov.txt"
    path.write_text(content)
    result = parse_triplet_cov(path)
    cov = result["cov"]
    assert result["index_base_detected"] == "1-based"
    assert np.allclose(cov, np.array([[4.0, 1.0], [1.0, 9.0]]))


def test_parse_triplet_cov_zero_based(tmp_path):
    content = """
0 0 4
0 1 1
1 1 9
"""
    path = tmp_path / "cov.txt"
    path.write_text(content)
    result = parse_triplet_cov(path)
    cov = result["cov"]
    assert result["index_base_detected"] == "0-based"
    assert np.allclose(cov, np.array([[4.0, 1.0], [1.0, 9.0]]))


def test_parse_triplet_cov_duplicate_consistent(tmp_path):
    content = """
1 1 4
1 2 1
2 1 1
2 2 9
"""
    path = tmp_path / "cov.txt"
    path.write_text(content)
    result = parse_triplet_cov(path)
    cov = result["cov"]
    assert np.allclose(cov, np.array([[4.0, 1.0], [1.0, 9.0]]))
