import medrag_settings


def test_package_imports() -> None:
    assert medrag_settings.__version__ == "0.1.0"
