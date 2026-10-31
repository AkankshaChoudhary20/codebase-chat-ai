import tempfile
import unittest
from pathlib import Path
from app.ingestion import iter_source_files
from app.chunking import chunk_file

class Utf8BomSourceTests(unittest.TestCase):
    def test_bom_source_keeps_function_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'example.py').write_bytes(b'\xef\xbb\xbfdef answer():\n    return 42\n')
            files = list(iter_source_files(root, 1000))
            self.assertEqual(files, [('example.py', 'def answer():\n    return 42\n')])
            chunks = chunk_file(*files[0])
            self.assertEqual(chunks[0].symbol_name, 'answer')
            self.assertEqual(chunks[0].symbol_type, 'function')
