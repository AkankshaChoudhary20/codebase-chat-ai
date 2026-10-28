import tempfile
import unittest
from pathlib import Path
from app.ingestion import iter_source_files


class SourceWalkTests(unittest.TestCase):
    def test_symlinked_files_and_directories_are_not_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'repo';root.mkdir()
            outside=base/'private';outside.mkdir()
            (outside/'secret.py').write_text('SECRET = 1')
            (root/'linked.py').symlink_to(outside/'secret.py')
            (root/'linked-dir').symlink_to(outside,target_is_directory=True)
            (root/'safe.py').write_text('value = 1')
            self.assertEqual(list(iter_source_files(root,1000)),[('safe.py','value = 1')])
    def test_dependencies_binary_and_oversized_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['node_modules','thing.egg-info','__pycache__']:
                target=root/name;target.mkdir();(target/'skip.py').write_text('skip')
            (root/'binary.py').write_bytes(b'abc\0def')
            (root/'large.py').write_text('x'*21)
            (root/'ok.py').write_text('ok')
            self.assertEqual(list(iter_source_files(root,20)),[('ok.py','ok')])

    def test_python_stubs_and_windows_scripts_are_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['client.pyi', 'window.pyw', 'types.PYI']:
                (root / name).write_text('def run(): ...')
            self.assertEqual({name for name, _ in iter_source_files(root, 1000)},
                             {'client.pyi', 'window.pyw', 'types.PYI'})
