#!/home/ubuntu/anaconda3/bin/python

from source.configurations import *
from source.tagCreator import createModelTag

""" Module responsible for the training of the regressor with data sets.
"""
@train_options
@click.command()
def train(modeloutputdirectory, conversion, lossfunction, traindatasetfile):
    """
    Train function
    Attributes:
        modeloutputdirectory -- path to desired store location of models
        conversion --  read values conversion method 
        lossfunction -- loss function for training
        traindatasetfile -- input data set for training
    """

    ### checking extensions of files
    checkExtension(traindatasetfile, "z")
    ### load data set and set parameters
    df, params = joblib.load(traindatasetfile)
    params['conversion'] = conversion
    params['lossfunction'] = lossfunction
    modelTag = createModelTag(params)
    ### create model with desired parameters
    model = RandomForestRegressor(max_features='sqrt',random_state=5,\
                n_estimators =10,n_jobs=4, verbose=2, criterion=lossfunction)
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
    modelFileName = modeloutputdirectory + "/" + modelTag + ".z"
    joblib.dump((model, params), modelFileName,compress=True ) 
    print("\n")

if __name__ == '__main__':
    train()
