from src.storage import LocalStorage


def test_local_roundtrip(tmp_path):
    s = LocalStorage(str(tmp_path))
    s.put_bytes("models/a/model.bin", b"hello")
    assert s.exists("models/a/model.bin")
    assert s.get_bytes("models/a/model.bin") == b"hello"
    assert "models/a/model.bin" in s.list("models/")
    assert not s.exists("models/missing")
