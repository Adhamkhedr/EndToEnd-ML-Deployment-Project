from setuptools import setup, find_packages

def parse_requirements(filename):
    """Reads requirements.txt and returns a clean list of dependencies, skipping empty lines and comments."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="mlappbuilder",
    version="0.1.0",
    author="Adham Khedr",
    author_email="Adhamkhedr@aucegypt.edu",
    description="A modular machine learning pipeline with training, packaging, and deployment capabilities.",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
