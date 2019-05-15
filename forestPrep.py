from sklearn.ensemble import RandomForestRegressor
from hiCOperations import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def predict(args): 
    df = pickle.load(open( SET_D+args.chrom+"_allWindows.p", "rb" ) )
    df = df.fillna(value=0)
    model = pickle.load(open( MODEL_D +args.chrom+"_model.p", "rb" ) )
    matrix ="4_02893.cool"
    ma = hm.hiCMatrix(CHUNK_D+matrix)
    mat = ma.matrix.todense()
    s = ma.cut_intervals[0][1]
    e = ma.cut_intervals[-1][2]
    matData = df.where(df['first'] >= s).where(df['first'] <=
            e).where(df['second'] >= s).where(df['second'] <= e)
    matData = matData.fillna(value=0)

    print(matData.shape)
    X = matData[matData.columns.difference(['first, second, target'])]
    y_true = matData['target']
    y_pred = model.predict(X)
    print(model.score(X,y_true))
    print(y_true.append(y_pred))
    
def predictTest(args):
    df = pickle.load(open( SET_D+args.chrom+"_test.p", "rb" ) )
    model = pickle.load(open( MODEL_D +args.chrom+"_model.p", "rb" ) )
    df = df.fillna(value=0)
    test_X = df[df.columns.difference(['first', 'second', 'target'])]
    print(test_X)
    test_y = df['target']
    y_pred = model.predict(test_X)
    test_y = test_y.to_frame()
    print(model.score(test_X,test_y))
    test_y['pred'] = y_pred
    print(test_y[:20])

def joinDataset(args):
    data = pd.DataFrame() 
    for f in os.listdir(SET_D):
        df = pickle.load(open( SET_D+f, "rb" ) )
        data = data.append(df)
    pickle.dump(data, open( SET_D +args.chrom+"_allWindows.p", "wb" ) )
    print(data.shape)


def splitDataset2(args):
    df = pickle.load(open( SET_D+args.chrom+"_allWindows.p", "rb" ) )
    df = df.set_index(['first','second'])
    df = df.sort_index()
    df = df.drop_duplicates(keep=False, inplace=False)
    # df = df.set_index(['first','second','chrom'])
    s = 70140000
    e = 100180000
    test = df.loc[(df.index.get_level_values('first') >= s)]
    test = test.loc[(test.index.get_level_values('first') <= e)]
    test = test.loc[(test.index.get_level_values('second') >= s)]
    test = test.loc[(test.index.get_level_values('second') <= e)]
    train = df.subtract(test)
    x = pd.concat([df, test])
    train = x.drop_duplicates(keep=False, inplace=False)
    print(df.shape)
    print(train.shape)
    print(test.shape)
    pickle.dump(train, open( SET_D +args.chrom+"_train.p", "wb" ) )
    pickle.dump(test, open( SET_D +args.chrom+"_test.p", "wb" ) )

def splitDataset(args):
    df = pickle.load(open( SET_D+args.chrom+"_allWindows.p", "rb" ) )
    train, test = train_test_split(df, test_size=0.2)
    pickle.dump(train, open( SET_D +args.chrom+"_train.p", "wb" ) )
    pickle.dump(test, open( SET_D +args.chrom+"_test.p", "wb" ) )

def startTraining(args):
    model = RandomForestRegressor(random_state=0,n_estimators=args.estimators,n_jobs=3, verbose=3)
    df = pickle.load(open( SET_D+args.chrom+"_train.p", "rb" ) )
    df = df.fillna(value=0)
    # print(df.isnull().values.any())
    train_X = df[df.columns.difference(['first', 'second', 'target'])]
    train_y = df['target']
    model.fit(train_X,train_y)
    pickle.dump(model, open( MODEL_D +args.chrom+"_model.p", "wb" ) )
    
def parseArguments(args=None):
    print(args)

    parser = argparse.ArgumentParser(description='HiC Prediction')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--action', '-a', choices=['train',
         'predict', 'join', 'split'], help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--learningRate', '-lr',type=float, default=0.001)
    parserOpt.add_argument('--estimators', '-e',type=int, default=20)
    parserOpt.add_argument('--treshold', '-tr',type=int, default=0)
    parserOpt.add_argument('--beta', '-be',type=float, default=0.1)
    parserOpt.add_argument('--chrom', '-c',type=str, default="4")
    parserOpt.add_argument('--cutLength', '-cl',type=int, default=50)
    parserOpt.add_argument('--maxValue', '-mv',type=int, default=10068)
    parserOpt.add_argument('--batchSize', '-b',type=int, default=256)
    parserOpt.add_argument('--model', '-m',type=str, default='autoencoder.pt')
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
        predictTest(args)
    elif args.action == "split":
        splitDataset2(args)
    elif args.action == "join":
        joinDataset(args)


if __name__ == "__main__":
    main(sys.argv[1:])
    # df = pickle.load(open( SET_D+"4_test.p", "rb" ) )
    # print(df)

