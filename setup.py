from setuptools import setup, find_packages
import versioneer

requirements = [
            'click',
            'cooler',
            'graphviz',
            'h5py',
            'hicexplorer',
            'hicmatrix',
            'hyperopt',
            'joblib',
            'matplotlib',
            'numpy', 
            'pandas',
            'pybedtools',
            'pyBigWig',
            'pydot',
            'scikit-learn',
            'scipy',
            'tqdm',    
]

setup(
    name='hicprediction',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="HiCMatrix Prediction via protein levels",
    author="Andre Bajorat, Ralf Krauth",
    url='https://github.com/MasterprojectRK/HiCPrediction',
    packages=find_packages(),
    include_package_data = True,
    package_data={'hicprediction': ['scripts/*.py']},
    install_requires=requirements,
    entry_points = {
        'console_scripts': ['createBaseFile=hicprediction.createBaseFile:loadAllProteins',
                            'createTrainingSet=hicprediction.createTrainingSet:createTrainingSet',
                            'training=hicprediction.training:train',
                            'predict=hicprediction.predict:executePredictionWrapper']
    },
    keywords='HiCPrediction',
)
