import pytest
from app.chunking import chunk_file


@pytest.mark.parametrize('extension', ['.pyi', '.pyw', '.PYI'])
def test_python_variants_have_searchable_symbol_chunks(extension):
    source = 'class Client:\n    def send(self, text: str) -> bool: ...\n'
    chunks = chunk_file('client' + extension, source)
    assert [(c.symbol_name, c.symbol_type) for c in chunks] == [
        ('Client', 'class'), ('Client.send', 'method')]
