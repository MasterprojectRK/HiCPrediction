from hiCOperations import *
from tagCreator import createModelTag

@click.option('--experimentOutputDirectory','-eod', required=True,\
              help='Outputfile')
@train_options
@click.argument('traindatasets', nargs=-1)
@click.command()
def train(experimentoutputdirectory, conversion, lossfunction, traindatasets ):
    if not os.path.isdir(experimentoutputdirectory + "/Models"):
        os.mkdir(experimentoutputdirectory + "/Models")
    print(traindatasets)
    for fileName in traindatasets:
        setTag = fileName.split("/")[-1].split(".")[0]
        modelTag = createModelTag(setTag, conversion, lossfunction)
        model = RandomForestRegressor(max_features='sqrt',random_state=5,\
                    n_estimators =10,n_jobs=4, verbose=2, criterion=lossfunction)
        df = pd.read_parquet(fileName)
        df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(value=0)
        X = df[df.columns.difference(['first', 'second','chrom', 'reads'])]
        if conversion == 'none':
            y = df['reads']
        elif conversion == 'standardLog':
            y = np.log(df['reads']+1)

        model.fit(X, y)
        modelFileName = experimentoutputdirectory + "/Models/" + modelTag +".z"
        joblib.dump(model, modelFileName,compress=True ) 

if __name__ == '__main__':
    train()
