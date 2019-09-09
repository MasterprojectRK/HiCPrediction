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
    # },
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
