#!/usr/bin/env python3

from hiCOperations import *
from tagCreator import createModelTag

@train_options
@click.command()
def train(modeloutputdirectory, conversion, lossfunction, traindatasetfile):
    checkExtension(traindatasetfile, "z")
    df, params = joblib.load(traindatasetfile)
    params['conversion'] = conversion
    params['lossfunction'] = lossfunction
    modelTag = createModelTag(params)
    model = RandomForestRegressor(max_features='sqrt',random_state=5,\
                n_estimators =10,n_jobs=4, verbose=2, criterion=lossfunction)
    df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(value=0)
    X = df[df.columns.difference(['first', 'second','chrom', 'reads'])]
    if conversion == 'none':
        y = df['reads']
    elif conversion == 'standardLog':
        y = np.log(df['reads']+1)

    model.fit(X, y)
    modelFileName = modeloutputdirectory + "/" + modelTag + ".z"
    joblib.dump((model, params), modelFileName,compress=True ) 
    print("\n")

if __name__ == '__main__':
    train()
