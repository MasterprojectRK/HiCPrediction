import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hiCPrediction-pkg-abajorat",
    version="0.0.1",
    author="Andre Bajorat",
    author_email="abajorat@posteo.de",
    description="PAckage to predict HiC-matricesbased on proteins",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abajorat/MasterProjekt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
