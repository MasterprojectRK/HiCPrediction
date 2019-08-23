import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hiCPrediction",
    version="0.0.1",
    author="Andre Bajorat",
    author_email="abajorat@posteo.de",
    description="Package to predict HiC-matrices based on proteins",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abajorat/MasterProjekt",
    packages=['InternalStorage','src'],
    package_data={'InternalStorage': ['centromeres.txt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    scripts=['src/createBaseFile.py', 'src/predict.py',
             'src/training.py', 'src/createTrainingSet.py',
            'src/createAllTrainSets.py','src/plotMatrix.py'],
     install_requires=[
          'joblib', 'pandas', 'future',
         'scikit-learn', 'unidecode','matplotlib', 'pyarrow'],


)
