from setuptools import setup, find_packages

setup(
    name="discover-infant",
    version="0.1.0",
    description="Discovering hidden visual concepts beyond linguistic input in Infant learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ke Xueyi",
    author_email="xke001@e.ntu.edu.sg",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)