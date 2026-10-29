import pytest
from app.chunking import chunk_file
from app.ingestion import iter_source_files

@pytest.mark.parametrize('extension', ['.mjs', '.cjs', '.mts', '.cts', '.MJS'])
def test_module_variants_are_walked_and_chunked_by_symbol(tmp_path, extension):
    name = 'worker' + extension
    source = 'export function run() {\n  return 42;\n}\n'
    (tmp_path / name).write_text(source)
    files = list(iter_source_files(tmp_path, 10000))
    assert files == [(name, source)]
    chunks = chunk_file(*files[0])
    assert [(c.symbol_name, c.symbol_type, c.start_line, c.end_line) for c in chunks] == [('run', 'function', 1, 3)]
