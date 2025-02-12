import os
from tempfile import TemporaryDirectory
from repomap.repomap import RepoMap
import sys
import tempfile

class IgnorantTemporaryDirectory:
    def __init__(self):
        if sys.version_info >= (3, 10):
            self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()

    def __enter__(self):
        return self.temp_dir.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        try:
            self.temp_dir.cleanup()
        except (OSError, PermissionError, RecursionError):
            pass  # Ignore errors (Windows and potential recursion)

    def __getattr__(self, item):
        return getattr(self.temp_dir, item)

def test_get_repo_map_excludes_added_files():
    # Create a temporary directory with sample files for testing
    test_files = [
        "test_file1.py",
        "test_file2.py",
        "test_file3.md",
        "test_file4.json",
    ]

    with IgnorantTemporaryDirectory() as temp_dir:
        for file in test_files:
            with open(os.path.join(temp_dir, file), "w") as f:
                f.write("def foo(): pass\n")

        repo_map = RepoMap(root=temp_dir, verbose=True)
        test_files = [os.path.join(temp_dir, file) for file in test_files]
        result = repo_map.get_repo_map(test_files[2:], test_files[:2])

        print(f"result >>>>>>>>>>>\n{result}\n<<<<<<<<<<<<")
        # dump(result)

        # Check if the result contains the expected tags map
        assert "test_file1.py" not in result
        assert "test_file2.py" not in result
        assert "test_file3.md" in result
        assert "test_file4.json" in result

        # close the open cache files, so Windows won't error
        del repo_map

test_get_repo_map_excludes_added_files()
