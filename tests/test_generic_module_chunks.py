import unittest

from app.chunking import chunk_file


class GenericModuleChunkTests(unittest.TestCase):
    def test_imports_constants_and_entrypoints_stay_searchable(self):
        sources = {
            'client.ts': 'import { send } from "./http";\nconst TIMEOUT = 30;\n\nexport function run() {\n  return send(TIMEOUT);\n}\n\nrun();\n',
            'main.go': 'package main\nimport "fmt"\nconst Timeout = 30\n\nfunc main() {\n    fmt.Println(Timeout)\n}\n',
            'lib.rs': 'use std::time::Duration;\nconst TIMEOUT: u64 = 30;\n\nfn run() {\n    work();\n}\n',
            'Main.java': 'package example;\nimport java.util.List;\n\npublic class Main {\n    public void run() {}\n}\n',
        }
        for path, source in sources.items():
            with self.subTest(path=path):
                chunks = chunk_file(path, source)
                lines = source.splitlines()
                covered = set()
                for chunk in chunks:
                    self.assertEqual(chunk.code_text, '\n'.join(lines[chunk.start_line - 1:chunk.end_line]))
                    covered.update(range(chunk.start_line, chunk.end_line + 1))
                for number, line in enumerate(lines, 1):
                    if line.strip():
                        self.assertIn(number, covered)
                self.assertTrue(any(chunk.symbol_type == 'module' for chunk in chunks))
                self.assertTrue(any(chunk.symbol_type != 'module' for chunk in chunks))

    def test_symbol_only_file_does_not_get_duplicate_module_chunk(self):
        chunks = chunk_file('x.js', '\nfunction run() {\n  return 1;\n}\n\n')
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, 'run')

    def test_symbol_free_file_keeps_whole_file_fallback(self):
        source = 'import "./setup";\nconst TIMEOUT = 30;\n'
        chunks = chunk_file('config.js', source)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].code_text, source)
