from app.chunking import chunk_file

def test_bom_prefixed_python_keeps_symbol_boundaries_and_line_numbers():
    source = '\ufeff# module\nVALUE = 1\ndef greet():\n    return VALUE\n'
    chunks = chunk_file('example.py', source)
    function = next(c for c in chunks if c.symbol_name == 'greet')
    assert (function.start_line, function.end_line) == (3, 4)
    assert function.code_text == 'def greet():\n    return VALUE'
    assert any('VALUE = 1' in c.code_text for c in chunks)
