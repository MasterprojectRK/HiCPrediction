from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc

from hicmatrix import HiCMatrix as hm
from hicexplorer import hicPlotMatrix as hicPlot

import h5py
import joblib
import sys
import bisect 
import argparse
import glob
import math
import time
import datetime
import itertools
import shutil
import operator
import subprocess
import click
import pickle
import os
import numpy as np
import logging as log
import pandas as pd
from copy import copy, deepcopy
from io import StringIO
from csv import writer
from tqdm import tqdm
import logging

import cooler
import pybedtools

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

from scipy import sparse
from scipy import signal
from scipy import misc
from scipy.stats.stats import pearsonr
from scipy.sparse import coo_matrix

log.basicConfig(level=log.DEBUG)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3, suppress=True)

class Mutex(click.Option):
    def __init__(self, *args, **kwargs):
        self.not_required_if:list = kwargs.pop("not_required_if")

        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs["help"] = (kwargs.get("help", "") + "Option is mutually exclusive with " + ", ".join(self.not_required_if) + ".").strip()
        super(Mutex, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        current_opt:bool = self.name in opts
        for mutex_opt in self.not_required_if:
            if mutex_opt in opts:
                if current_opt:
                    raise click.UsageError("Illegal usage: '" + str(self.name)\
                                           +"' is mutually exclusive with " + str(mutex_opt) + ".")
                else:
                    self.prompt = None
        return super(Mutex, self).handle_parse_result(ctx, opts, args)

_setAndProtein_options = [
    click.option('--baseFile','-bf', required=True,type=click.Path(writable=True),
        help='Base file where to store proteins and chromosomes for later use.'),
    click.option('--chromosomes', '-chs', default=None, help=\
                "If set, sets are only calculated for these chromosomes instead of all"),
]
_predict_options = [
    click.option('--predictionOutputDirectory', '-dod',required=True,type=click.Path(exists=True),\
                 help='Output directory for training set files'),
    click.option('--modelFilePath', '-mfp', required=True,\
              help='Choose model on which to predict'),
    click.option('--baseFile','-bf', required=True,type=click.Path(writable=True),
        help='Base file from where to load chromosomes.'),
]

_protein_options = [
    click.option('--resolution' ,'-r', required=True,\
                help = "Store resolution for analys and documentation"),
    click.option('--cellType' ,'-ct', required=True, \
                help="Store cell type for analysis and documentation"),
    click.option('--matrixFile', '-mf',required=True,type=click.Path(exists=True),\
                 help='Input file with the whole HiC-matrix ')
]

_set_options = [
    click.option('--peakColumn' ,'-pc', default=6,hidden=True),
    click.option('--mergeOperation','-mo',default='avg',\
                 type=click.Choice(['avg', 'max']),show_default=True,\
                 help='This parameter defines how the proteins are binned'),
    click.option('--normalize', default=False,\
                 show_default=True,\
                 help='Should the proteins be normalized to a 0-1 range'),
    click.option('--equalize', default=False,
                 show_default=True,hidden=True,
                help='If either of the basepairs has no peak at a specific '+\
                'protein, set both values to 0'),
    click.option('--ignoreCentromeres', default=True,\
                 show_default=True,help='Cut out the centroid arms for training'),
    click.option('--windowOperation', '-wo', default='avg',\
              type=click.Choice(['avg', 'max', 'sum']), show_default=True,\
                help='How should the proteins in between two base pairs be summed up'),
    click.option('--windowSize', '-ws', default=200, show_default=True,\
                help='Maximum distance between two basepairs'),
    click.option('--centromeresFile', '-cmf',default='Data/centromeres.txt',\
              type=click.Path(exists=True)),
    click.option('--datasetOutputDirectory', '-dod',required=True,type=click.Path(exists=True),\
                 help='Output directory for training set files')
]
_train_options = [
    click.option('--lossfunction', '-lf', default='mse',\
              type=click.Choice(['mse','mae']), show_default=True, \
                help='Which loss function should be used for training'),
    click.option('--conversion', '-co', default='none',\
              type=click.Choice(['standardLog', 'none']), show_default=True,\
                help='Define a conversion function for the read values'),
    click.option('--trainDatasetFile', '-sof',\
                 required=True,\
                 help='File from which training is loaded'\
                 ,type=click.Path(writable=True)),
    click.option('--modelOutputDirectory', '-mod',required=True,type=click.Path(exists=True),\
                 help='Output directory for model files')
]


def protein_options(func):
    for option in reversed(_protein_options):
        func = option(func)
    for option in reversed(_setAndProtein_options):
        func = option(func)
    return func

def set_options(func):
    for option in reversed(_set_options):
        func = option(func)
    for option in reversed(_setAndProtein_options):
        func = option(func)
    return func

def predict_options(func):
    for option in reversed(_predict_options):
        func = option(func)
    return func
def train_options(func):
    for option in reversed(_train_options):
        func = option(func)
    return func

from itertools import product

def getCombinations():
    params = {
        'mergeOperation': ["avg", "max"],
        'normalize': [True, False],
        'peakColumn': [4, 6],
    }

    keys = list(params)
    paramDict =  product(*params.values())
    for val in tqdm(list(paramDict), desc= 'Iterate parameter combinations' ):
        yield dict(zip(params, val))


def checkExtension(fileName, extension, option=None):
    if fileName.split(".")[-1] != extension:
        if option and fileName.split(".")[-1] == option:
            return
        else:
            msg = 'The file {} has the wrong extension. Ensure to '\
                    +'pass a file with .{} extension'
            print(msg.format(str(fileName), extension))
            sys.exit()

