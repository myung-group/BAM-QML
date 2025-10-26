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
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pennylane==0.42.3",
        "jax==0.6.2",
        "jaxlib==0.6.2",
        "matscipy==1.1.1",
        "ase==3.23.0",
        "optax==0.2.5",
        "flax==0.11.2",
        "jraph==0.0.6.dev0",
        "e3nn-jax==0.20.8",
        "dm-haiku==0.0.14"
    ],
)



