def test_python_version():
    import sys
    assert sys.version_info >= (3, 11), "Python version needs to be at least 3.11"
