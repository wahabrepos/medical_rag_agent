import medrag_api


def test_package_imports() -> None:
    assert medrag_api.__version__ == "0.1.0"
