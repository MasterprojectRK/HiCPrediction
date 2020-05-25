from setuptools import setup, find_packages
import versioneer

requirements = [
            'bedtools>=2.29.2',
            'click>=7.0',
            'cooler>=0.8.7',
            'graphviz>=2.42.3',
            'h5py>=2.10.0',
            'hicexplorer>=3.4.1',
            'hicmatrix>=11',
            'hyperopt>=0.2.4',
            'joblib>=0.14.1',
            'matplotlib>=3.1.3'
            'numpy>=1.18.1' 
            'pandas>=1.0.1',
            'pybedtools>=0.8.0',
            'pybigwig>=0.3.17',
            'pydot>=1.4.1'
            'scikit-learn>=0.22.1',
            'scipy>=1.4.1'
            'tqdm>=4.43.0',    
]

setup(
    name='HiCPrediction',
    version="0.0.2",
    cmdclass=versioneer.get_cmdclass(),
    description="HiCMatrix Prediction via protein levels",
    author="Andre Bajorat, Ralf Krauth",
    url='https://github.com/MasterprojectRK/HiCPrediction',
    packages=find_packages(where='hicprediction'),
    package_data={'hicprediction': ['hicprediction/scripts/*.py']},

    scripts=[
             'hicprediction/createBaseFile.py',
             'hicprediction/predict.py',
             'hicprediction/training.py',
             'hicprediction/createTrainingSet.py',
            ],
    install_requires=requirements,
    keywords='HiCPrediction',
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
