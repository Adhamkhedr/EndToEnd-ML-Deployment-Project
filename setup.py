from setuptools import setup, find_packages

setup(
    name="mlappbuilder",
    version="0.1.0",
    author="Adham Khedr",
    author_email="Adhamkhedr@aucegypt.edu",
    description="A modular machine learning pipeline with training, packaging, and deployment capabilities.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "flask"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
