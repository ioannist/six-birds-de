import pandas as pd

from sixbirds_cosmo.datasets.des_y3_chain_meta import (
    extract_bestfit_row,
    parse_chain_header_comments,
    parse_paramnames_file,
)


def test_parse_paramnames_one_per_line(tmp_path):
    p = tmp_path / "params.txt"
    p.write_text("om\nob\nloglike\n")
    names = parse_paramnames_file(p)
    assert names == ["om", "ob", "loglike"]


def test_parse_header_comments_extracts_names(tmp_path):
    p = tmp_path / "chain.txt"
    p.write_text("# a b loglike\n1 2 3\n")
    out = parse_chain_header_comments(p)
    assert out["parsed_names"] == ["a", "b", "loglike"]


def test_extract_bestfit_by_loglike():
    df = pd.DataFrame({"a": [0.0, 1.0], "loglike": [-2.0, -1.0]})
    info = {"loglike": "loglike"}
    best = extract_bestfit_row(df, info=info)
    assert best["row_index"] == 1
    assert best["criterion"] == "loglike"


def test_no_invention_when_names_missing(tmp_path):
    # parse_chain_header_comments returns none; ensure parsed_names empty
    p = tmp_path / "chain.txt"
    p.write_text("1 2 3\\n")
    out = parse_chain_header_comments(p)
    assert out["parsed_names"] == []
