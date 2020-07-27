import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cell_models", # Replace with your own username
    version="0.0.1",
    author="Alex Clark",
    author_email="apclarkva@gmail.com",
    description="This package contains cell models and the ability to apply various protocols to them.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={'cell_models': ['ga/*.npy']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
