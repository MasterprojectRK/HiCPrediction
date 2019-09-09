#!/usr/bin/env python3

from hicprediction.training import training
from  hicprediction.configurations import *
import json

@alltrain_options
@click.command()
def create(setdirectory,conversion, modeloutputdirectory):
    for path in tqdm(os.listdir(setdirectory),\
                     desc="Iterating all of the training sets in directory"):
        training(modeloutputdirectory, conversion , setdirectory\
              +"/" +path)

if __name__ == '__main__':
    create()
