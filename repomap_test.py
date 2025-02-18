import os
from tempfile import TemporaryDirectory
from repomap.repomap import RepoMap
import sys
import tempfile
from tree_sitter_language_pack import get_language, get_parser
import json
from repomap.repomap import node_to_dict
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

ignore_directories = [".git", "node_modules", "ios", "android", "ios/build", "android/build", "windows", "linux", "macos", "web", "linux/build", "macos/build", "web/build"]
ignore_files = [".gitignore", "METADATA", "package.json", "package-lock.json", "package.yaml", "package.lockb", "package.lockb.json", "package.lockb.yaml", "package.lockb.toml"]
ignore_extensions = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".xml", ".css", ".js", ".jsx", ".ts", ".md", ".txt", ".log", ".yml", ".yaml", ".toml", ".lock", ".lockb", ".lockb.json", ".lockb.yaml", ".lockb.toml", ".ttf", ".json", ".lock", ".arb", ".yaml.lock", ".bat"]

def get_all_files(project_dir):
    all_files = []
    for root, dirs, files in os.walk(project_dir):
        # Remove directories that should be ignored so that os.walk doesn't traverse them.
        dirs[:] = [d for d in dirs if d not in ignore_directories]

        for file in files:
            if file in ignore_files:
                continue
            if os.path.splitext(file)[1] in ignore_extensions:
                continue
            all_files.append(os.path.join(root, file))
    return all_files

def test_get_repo_map_excludes_added_files():
    # list all files from the project directory recursively
    project_dir = "/Users/burnerlee/Projects/random/scratch-workspace-flutter"
    files = get_all_files(project_dir)

    for file in files:
        print(f"file: {file}")
    # print(f"files: {files}")
    # dump(result)

    # Check if the result contains the expected tags map
    repo_map = RepoMap(project_dir)
    result = repo_map.get_repo_map([], files)
    token = repo_map.token_count(result)
    print(f"result: {result}")
    print(f"token: {token}")
    # close the open cache files, so Windows won't error
    del repo_map

test_get_repo_map_excludes_added_files()

def test_dart_tags():
    main_dart_file = os.path.join(os.path.dirname(__file__), "main.dart")
    file_content = open(main_dart_file).read()

    parser = get_parser("dart")
    language = get_language("dart")
    tree = parser.parse(bytes(file_content, "utf-8"))

    # tree_dict = node_to_dict(tree.root_node)
    # print(json.dumps(tree_dict, indent=2))
    query_scm = open(os.path.join(os.path.dirname(__file__), "repomap", "queries", "tree-sitter-dart-tags.scm")).read()

    # Run the tags queries
    query = language.query(query_scm)
    captures = query.captures(tree.root_node) 
    print(captures)

# test_dart_tags()
