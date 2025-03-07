from setuptools import setup, find_packages

setup(
    name="repomap",
    version="0.1.1",
    packages=find_packages(),
    include_package_data=True,  # Include non-Python files (if any)
    install_requires=[
        "tree-sitter-language-pack",
        "pygments",
        "networkx",
        "tiktoken",
        "tqdm",
        "tree-sitter==0.24.0",
    ],
    package_data={
        "repomap": ["vendor/**/*", "queries/**/*"],  # Ensure vendor files are included
    },
)
