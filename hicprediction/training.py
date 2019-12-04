#!/usr/bin/env python3

from hicprediction.configurations import *
from hicprediction.tagCreator import createModelTag

""" Module responsible for the training of the regressor with data sets.
"""
@train_options
@click.command()
def train(modeloutputdirectory, conversion, traindatasetfile):
    """
    Wrapper function for click
    """
    training(modeloutputdirectory, conversion, traindatasetfile)

def training(modeloutputdirectory, conversion, traindatasetfile):
    """
    Train function
    Attributes:
        modeloutputdirectory -- path to desired store location of models
        conversion --  read values conversion method 
        traindatasetfile -- input data set for training
    """
    ### checking extensions of files
    checkExtension(traindatasetfile, "z")
    ### load data set and set parameters
    df, params = joblib.load(traindatasetfile)
    params['conversion'] = conversion
    modelTag = createModelTag(params) + ".z"
    modelFileName = os.path.join(modeloutputdirectory, modelTag)
    if not os.path.isfile(modelFileName):
        ### create model with desired parameters
        model = RandomForestRegressor(max_features='sqrt',random_state=5,\
                    n_estimators =10,n_jobs=4, verbose=2, criterion='mse')
        df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(value=0)
        ### eliminate columns that should not be used for training
        X = df[df.columns.difference(['first', 'second','chrom', 'reads', 'avgRead'])]
        ### apply conversion
        if conversion == 'none':
            y = df['reads']
        elif conversion == 'standardLog':
            y = np.log(df['reads']+1)

        ## train model and store it
        model.fit(X, y)
        joblib.dump((model, params), modelFileName,compress=True ) 
        print("\n")
    else:

        print("Skipped creating model that already existed: " + modelFileName)

if __name__ == '__main__':
    train()
