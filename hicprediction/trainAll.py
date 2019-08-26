#!/usr/bin/env python3

from hicprediction.training import train
from  hicprediction.configurations import *
import json

@alltrain_options
@click.command()
def createAllModels(setdirectory,conversion,lossfunction, modeloutputdirectory):
    for path in os.listdir(setdirectory):
        train(modeloutputdirectory, conversion , lossfunction, setdirectory\
              +"/" +path)

if __name__ == '__main__':
    createAllModels()
