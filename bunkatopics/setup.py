from distutils.core import setup

setup(
    name="bunkatopics",
    packages=["bunkatopics"],
    version="0.1",
    license="MIT",
    description="Advanced Topic Modeling using transformers",
    author="Charles de Dampierre",
    author_email="charles.de-dampierre@hec.edu",
    url="https://github.com/charlesdedampierre/BunkaTopics",
    download_url="https://github.com/charlesdedampierre/BunkaTopics/archive/v_01.tar.gz",
    keywords=[
        "Topic Modeling",
        "NLP",
        "Search",
    ],
    install_requires=[
        "validators",
        "beautifulsoup4",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
