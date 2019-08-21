import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hiCPrediction-abajorat",
    version="0.0.1",
    author="Andre Bajorat",
    author_email="abajorat@posteo.de",
    description="Package to predict HiC-matrices based on proteins",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abajorat/MasterProjekt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    scripts=['HiCPrediction/createBaseFile.py', 'HiCPrediction/predict',
             'HiCPrediction/training.py', 'HiCPrediction/createTrainingSet.py',
            'HiCPrediction/plotMatrix.py'],
)
