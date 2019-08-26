from setuptools import setup
import versioneer

requirements = [
            'joblib', 
            'pandas',
            'future',
            'scikit-learn',
            'unidecode',
            'matplotlib',
            'pyarrow',  
            # 'hic2cool >=0.7',
            # 'pybedtools',
            # 'pybigwig',
            # 'hicmatrix >= 11',
            # 'hicexplorer',
]

setup(
    name='HiCPrediction',
    version="0.0.1",
    cmdclass=versioneer.get_cmdclass(),
    description="HiCMatrix Prediction via protein levels",
    author="Andre Bajorat",
    author_email='abajorat@posteo.de',
    url='https://github.com/abajorat/HiCPrediction',
    packages=['hicprediction'],
    package_data={'hicprediction': ['InternalStorage/centromeres.txt']},
    # entry_points={
        # 'console_scripts': [
            # 'src=src.cli:cli'
        # ]
    # },
    scripts=['hicprediction/createBaseFile.py', 'hicprediction/trainAll.py','hicprediction/predict.py',
             'hicprediction/training.py', 'hicprediction/createTrainingSet.py',
            'hicprediction/createAllTrainSets.py','hicprediction/plotMatrix.py'],
    install_requires=requirements,
    keywords='HiCPrediction',
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
