from hiCOperations import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

def predict(args):
    if not os.path.exists(args.predDir):
        os.mkdir(args.predDir)
    df = pickle.load(open( args.sourceFile, "rb" ) )
    model = pickle.load(open( args.modelFile, "rb" ) )
    df = df.fillna(value=0)
    test_X = df[df.columns.difference(['first', 'second', 'target'])]
    print(test_X)
    print(test_X)
    test_y = df['target']
    y_pred = model.predict(test_X)
    test_y = test_y.to_frame()
    print(model.score(test_X,test_y))
    test_y['first'] = df['first']
    test_y['second'] = df['second']
    test_y['pred'] = y_pred
    test_y = test_y.set_index(['first','second'])
    # print(test_y.loc[(161480000,161500000)])

    pickle.dump(test_y, open(args.predDir+"/" PREDTYPE, "wb" ) )
    

def createCombinedDataset(args):
    d = SET_D +args.conversion+"/all/"
    chroms = [item for item in args.chroms.split('_')]
    data = pd.DataFrame() 
    for f in chroms:
        df = pickle.load(open(d+f+".p", "rb" ) )
        data = data.append(df)
    pickle.dump(data, open(SET_D+args.conversion+"/combined/"+args.chroms+".p", "wb" ) )
    print(data.shape)


def splitDataset2(args):
    df = pickle.load(open( SET_D+"all/"+args.chrom+"_allWindows.p", "rb" ) )
    df = df.set_index(['first','second'])
    df = df.sort_index()
    df = df.drop_duplicates(keep=False, inplace=False)
    df = df.fillna(value=0)
    # df = df.set_index(['first','second','chrom'])
    s = 70140000
    e = 100180000
    test = df.loc[(df.index.get_level_values('first') >= s)]
    test = test.loc[(test.index.get_level_values('first') <= e)]
    test = test.loc[(test.index.get_level_values('second') >= s)]
    test = test.loc[(test.index.get_level_values('second') <= e)]
    print(test)
    # train = df.subtract(test)
    x = pd.concat([df, test])
    train = x.drop_duplicates(keep=False, inplace=False)
    print(df.shape)
    print(train.shape)
    print(test.shape)
    pickle.dump(train, open( SET_D+"train/" +args.chrom+"_train.p", "wb" ) )
    pickle.dump(test, open( SET_D +"test/"+args.chrom+"_test.p", "wb" ) )

def splitDataset(args):
    df = pickle.load(open( SET_D+args.chrom+"_allWindows.p", "rb" ) )
    train, test = train_test_split(df, test_size=0.2)
    pickle.dump(train, open( SET_D +args.chrom+"_train.p", "wb" ) )
    pickle.dump(test, open( SET_D +args.chrom+"_test.p", "wb" ) )

def startTraining(args):
    model = RandomForestRegressor(random_state=5,n_estimators=args.estimators,n_jobs=3, verbose=3)
    df = pickle.load(open( args.trainFile, "rb" ) )
    # print(df.isnull().values.any())
    df.replace([np.inf, -np.inf], np.nan)
    print(df.max())
    print(df)
    df = df.fillna(value=0)
    X = df[df.columns.difference(['first', 'second', 'target'])]
    y = df['target']
    print(type(y[0]))
    model.fit(X, y)
    pickle.dump(model, open(args.modelFile, "wb" ) )
    
def parseArguments(args=None):
    print(args)

    parser = argparse.ArgumentParser(description='HiC Prediction')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--action', '-a', choices=['train',
         'predict','combine', 'split'], help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--learningRate', '-lr',type=float, default=0.001)
    parserOpt.add_argument('--estimators', '-e',type=int, default=20)
    parserOpt.add_argument('--trainFile', '-tf',type=str, default="")
    parserOpt.add_argument('--sourceFile', '-sf',type=str, default="")
    parserOpt.add_argument('--modelFile', '-mf',type=str, default="")
    parserOpt.add_argument('--predDir', '-pd',type=str, default="")
    parserOpt.add_argument('--chrom', '-c',type=str, default="4")
    parserOpt.add_argument('--chroms', '-cs',type=str, default="1_2")
    parserOpt.add_argument('--cutLength', '-cl',type=int, default=50)
    parserOpt.add_argument('--maxValue', '-mv',type=int, default=10068)
    parserOpt.add_argument('--batchSize', '-b',type=int, default=256)
    parserOpt.add_argument('--model', '-m',type=str, default='autoencoder.pt')
    parserOpt.add_argument('--conversion', '-co',type=str, default="default")
    parserOpt.add_argument('--saveModel', '-sm', default=True)
    parserOpt.add_argument('--loss', '-l',type=str, default='L1')
    parserOpt.add_argument('--prepData', '-dp',type=str, default='log')
    parserOpt.add_argument('--validationData', '-v',type=bool, default=False)
    parserOpt.add_argument('--trainData', '-t',type=bool,default=True)
    args = parser.parse_args(args)
    print(args)
    return args

def main(args=None):
    args = parseArguments(args)
    if args.action == "train":
        startTraining(args)
    elif args.action == "predict":
        predict(args)
    elif args.action == "split":
        splitDataset2(args)
    elif args.action == "combine":
        createCombinedDataset(args)


if __name__ == "__main__":
    main(sys.argv[1:])
    # df = pickle.load(open( SET_D+"4_test.p", "rb" ) )
    # print(df)

