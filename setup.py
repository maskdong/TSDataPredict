from setuptools import setup, find_packages

setup(
    name='timeseries-prediction-project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'torch',
        'notebook',
        'pyyaml'
    ],
)
