from setuptools import setup, find_packages

setup(
    name="RRL-Code",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "SimulationFramework @ git+http://alr_jacob@git.informatik.kit.edu/i53/SimulationFramework-fork@master",
        "tensorflow-gpu==2.2.0",
        "numpy==1.17.2",
        "gym==0.14.0",
        "dm_control",
        "tensorflow_probability==0.10.0",
        "pandas"
    ],
    # extra requirements only needed for development
    extras_require={
        "dev": [
            "pytest>=5.4.2"
        ]
    }
)
