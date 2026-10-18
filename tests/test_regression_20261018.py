import unittest
from app.chunking import chunk_python_source

class ModuleStatementTests(unittest.TestCase):
    def test_imports_constants_and_entrypoint_remain_searchable(self):
        source = 'import os\nTIMEOUT = 30\n\ndef run():\n    return TIMEOUT\n\nif __name__ == "__main__":\n    run()\n'
        chunks = chunk_python_source(source, 'example.py')
        indexed = '\n'.join(chunk.code_text for chunk in chunks)
        for expected in ['import os', 'TIMEOUT = 30', 'def run():', 'if __name__']:
            self.assertIn(expected, indexed)
        lines = source.splitlines()
        for chunk in chunks:
            self.assertEqual(chunk.code_text, '\n'.join(lines[chunk.start_line - 1:chunk.end_line]))

    def test_decorated_function_stays_intact(self):
        chunks = chunk_python_source('@cache\ndef run():\n    return 1', 'x.py')
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_line, 1)
