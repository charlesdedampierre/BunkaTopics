from setuptools import find_packages, setup

dependencies = [
    "pandas>=2.0.2",
    "umap-learn>=0.5.3",
    "pydantic>=2.5.3",
    "loguru>=0.7.0",
    "langchain>=0.0.206",
    "plotly>=5.15.0",
    "textacy>=0.13.0",
    "gensim>=4.3.1",
    "sentence-transformers>=2.7.0",
    "openai>=0.28.0",
    "python-dotenv>=1.0.0",
    "matplotlib>=3.7.2",
    "datasets>=2.14.5",
    "psutil>=5.9.7",
    "colorlog>=6.8.0",
    "langchain_openai",
    "ipython",
    "ipywidgets>=8.1.2",
    "jsonlines>=4.0.0",
    "pyod>=1.1.3",
    "FlagEmbedding>=1.2.8",
    "tiktoken==0.6.0",
    "langdetect>=1.0.9",
]

test = ["nbformat>=4.2.0", "nbconvert>=7.16.3", "jupyter>=1.00"]

format_dependencies = [
    "black ~= 23.0",
    "isort ~= 5.0",
    "twine ~= 4.0",
    "wheel",
    "kaleido",
    "flake8",
]

docs_dependencies = [
    "mkdocs>=1.1.2",
    "mkdocs-material>=8.1.4",
    "mkdocstrings>=0.24.0",
    "mkdocstrings-python>=1.8.0",
]

dev = test + format_dependencies + docs_dependencies

front = ["streamlit"]

with open("README.md", "r") as doc:
    long_description = doc.read()

setup(
    name="bunkatopics",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.46",
    author="Charles de Dampierre",
    author_email="charlesdedampierre@gmail.com",
    description="Bunkatopics is a Topic Modeling package and Exploration Module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/charlesdedampierre/BunkaTopics",
    project_urls={
        "Source code": "https://github.com/charlesdedampierre/BunkaTopics",
        "Documentation": "https://charlesdedampierre.github.io/BunkaTopics/",
        "Issue Tracker": "https://github.com/charlesdedampierre/BunkaTopics/issues",
    },
    keywords="AI Topic Modeling Visualization Exploration Fine-tuning",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=dependencies,
    extras_require={
        "dev": dev,
        "test": test,
        "docs": docs_dependencies,
        "format": format_dependencies,
    },
    python_requires=">=3.9",
)
