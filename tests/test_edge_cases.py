from app.chunking import chunk_file

def test_python_comments_outside_symbols_remain_searchable():
    source = "# Retry policy: exponential backoff\n\ndef retry():\n    pass\n\n# Operational shutdown guidance\n"
    chunks = chunk_file('retry.py', source)
    assert any('Retry policy' in c.code_text and c.start_line == 1 for c in chunks)
    assert any('Operational shutdown' in c.code_text for c in chunks)
    assert len([c for c in chunks if c.symbol_type == 'function']) == 1
