import subprocess
from app.ingestion import clone_repo

def test_repository_name_starting_with_dash_is_cloned_as_a_path(tmp_path, monkeypatch):
    source = tmp_path / "--bare.git"
    subprocess.run(["git", "init", "--bare", str(source)], check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    clone = clone_repo("--bare.git", str(tmp_path / "clones"))
    assert (clone / ".git").is_dir()
