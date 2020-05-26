from setuptools import setup, find_packages

setup(
    name="RRL-Code",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "SimulationFramework @ git+http://alr_jacob@git.informatik.kit.edu/i53/SimulationFramework-fork@master"
    ],
    # extra requirements only needed for development
    extras_require={
        "dev": [
            "pytest>=5.4.2"
        ]
    }
)
