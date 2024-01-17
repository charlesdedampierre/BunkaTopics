from setuptools import setup, find_packages

test_packages = ["pytest>=5.4.3", "pytest-cov>=2.6.1"]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
]

base_packages = [
    "pandas>=2.0.2",
    "umap-learn>=0.5.3",
    "pandas>=2.0.2",
    "pydantic>=1.10.9",
    "loguru>=0.7.0",
    "langchain>=0.0.206",
    "plotly>=5.15.0",
    "textacy>=0.13.0",
    "gensim>=4.3.1",
    # "torch>=2.0.1"
    "sentence-transformers>=2.2.2",
    "openai>=0.28.0",
    "python-dotenv>=1.0.0",
    "matplotlib>=3.7.2",
    "datasets>=2.14.5",
    "chromadb=0.4.13",
]

front_packages = ["streamlit>=1.26.0"]


dev_packages = docs_packages + test_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bunkatopics",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.43",
    author="charlesdedampierre",
    author_email="charlesdedampierre@gmail.com",
    description="Advanced Topic Visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/charlesdedampierre/BunkaTopics",
    project_urls={
        "Documentation": "https://charlesdedampierre.github.io/BunkaTopics/",
        "Source Code": "https://github.com/charlesdedampierre/BunkaTopics",
        "Issue Tracker": "https://github.com/charlesdedampierre/BunkaTopics/issues",
    },
    keywords="Artificial Intelligence generaive AI topic modeling embeddings",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "dev": dev_packages,
    },
    python_requires=">=3.9",
)
