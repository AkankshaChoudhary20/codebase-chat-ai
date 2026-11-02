from app.ingestion import iter_source_files

def test_source_newlines_are_normalized_before_chunking(tmp_path):
    source = tmp_path / 'source.py'
    source.write_bytes(b'def one():\r\n    pass\rdef two():\n    pass\n')
    assert list(iter_source_files(tmp_path, 1000)) == [('source.py', 'def one():\n    pass\ndef two():\n    pass\n')]
