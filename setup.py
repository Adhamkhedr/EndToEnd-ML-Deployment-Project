from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mlproject",
    version="0.1.0",
    author="Adham Khedr",
    author_email="Adhamkhedr@aucegypt.edu",
    description="A modular machine learning pipeline with training, packaging, and deployment capabilities.",
    packages=find_packages(where="src"),  #Look for packages inside the src/ directory, not the root directory.
    package_dir={"": "src"},              #When installing or importing, the root of my Python packages starts in src/
# Without these two lines , Setuptools would look in the root folder (./) and wouldn’t find code inside src/. 
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
