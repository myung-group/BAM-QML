from setuptools import find_packages, setup

setup(
    name="BAM-QML",
    version="0.1.0",
    author="Soohaeng Yoo Willow",
    author_email="sy7willow@gmail.com",
    description="Hybrid Quantum-Classical Machine Learning Potential with Variational Quantum Circuits",
    url="https://github.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        #Operating System :: OS Independent",
    ],
    install_requires=[
        "optax",
        "flax",
        "ase",
        "matscipy",
        "jraph",
        "e3nn-jax",
        "dm-haiku",
        "pennylane"
    ],
)



