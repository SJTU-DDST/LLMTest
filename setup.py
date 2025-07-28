from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="LLMTest",
    version="0.1.0",

    description="an easy-to-use LLM test framework",

    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/SJTU-DDST/LLMTest",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],

    python_requires='>=3.9',

    # install_requires=[
    #     "requests>=2.25.1",
    #     "numpy",
    # ],
)