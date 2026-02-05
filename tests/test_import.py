import sixbirds_cosmo


def test_import_and_version() -> None:
    assert hasattr(sixbirds_cosmo, "__version__")
