from hiCOperations import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

def predict(args):
    ma = hm.hiCMatrix(ARM_D +args.chrom+".cool")
    matrix = ma.matrix.todense()
    maxV = np.max(matrix)
    df = pickle.load(open( tagCreator(args, "set"), "rb" ) )
    model = pickle.load(open( tagCreator(args, "model"), "rb" ) )
    df = df.fillna(value=0)
    test_X = df[df.columns.difference(['first', 'second','chrom','reads', 'logTarget',
                                       'normTarget'])]
    test_y = df['reads']
    test_y = test_y.to_frame()
    test_y['normTarget'] = df['normTarget']
    test_y['logTarget'] = df['logTarget']
    y_pred = model.predict(test_X)
    test_y['pred'] = y_pred
    y_pred = np.absolute(y_pred)
    test_y['second'] = df['second']
    test_y['first'] = df['first']
    test_y['distance'] = df['distance']
    test_y['predAbs'] = y_pred
    if args.conversion == "norm":
        target = 'normTarget'
        reads = y_pred * maxV
    elif args.conversion == "log":
        target = 'logTarget'
        reads = y_pred * np.log(maxV)
        reads = np.exp(reads) - 1
    elif args.conversion == 'default':
        target = 'reads'
        reads = y_pred
    test_y['reads'] = df['reads']
    test_y['predReads'] = reads
    print(test_y)
    print(model.score(test_X,test_y[target]))
    test_y = test_y.set_index(['first','second'])
    if args.directConversion:
        predictionToMatrix(args, test_y) 
    else:
        pickle.dump(test_y, open(tagCreator(args, "pred"), "wb" ) )

def createCombinedDataset(args):
    chroms = [item for item in chromStringToList(args.chroms)]
    data = pd.DataFrame() 
    for f in chroms:
        print(f)
        if  "A" in args.arms:
            args.chrom = f + "_A"
            df = pickle.load(open(tagCreator(args, "set"), "rb" ) )
            data = data.append(df)
        if  "B" in args.arms:
            args.chrom = f + "_B"
            if  os.path.isfile(tagCreator(args, "set")):
                df = pickle.load(open(tagCreator(args, "set"), "rb" ) )
                data = data.append(df)
    pickle.dump(data, open(tagCreator(args, "setC"), "wb" ), protocol=4 )
    print(data.shape)



def startTraining(args):
    if args.model == "rf":

        model = RandomForestRegressor(random_state=5,n_estimators=args.estimators,
                                      n_jobs=3, verbose=3)
    elif args.model ==  "ada":
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'verbose':True,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    elif args.model ==  "mlp":
        model = MLPRegressor(hidden_layer_sizes=(20,10,5),verbose=True,
                             early_stopping=True, max_iter=25)
    path =  tagCreator(args, "setC")
    if not os.path.isfile(path):
        createCombinedDataset(args)

    df = pickle.load(open(path, "rb" ) )
    # print(df.isnull().values.any())
    df.replace([np.inf, -np.inf], np.nan)
    print(df.max())
    print(df)
    df = df.fillna(value=0)
    X = df[df.columns.difference(['first', 'second','chrom',
                                  'reads','logTarget','normTarget'])]
    if args.conversion == 'default':
        y = df['reads']
    else:
        y = df[args.conversion +'Target']
    print(type(y[0]))
    model.fit(X, y)
    pickle.dump(model, open(tagCreator(args, "model"), "wb" ) )
    
def parseArguments(args=None):
    print(args)

    parser = argparse.ArgumentParser(description='HiC Prediction')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--action', '-a', choices=['train',
         'predict','combine', 'split', 'createAllWindows','plotAll',
    'loadProtein','plot','plotPred','createArms', 'loadAllProteins'
                                                           ,  'createWindows' ], help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--estimators', '-e',type=int, default=20)
    parserOpt.add_argument('--sourceFile', '-sf',type=str, default="")
    parserOpt.add_argument('--chrom', '-c',type=str, default="4")
    parserOpt.add_argument('--reach', '-r',type=str, default="99")
    parserOpt.add_argument('--chroms', '-cs',type=str, default="1_2")
    parserOpt.add_argument('--arms', '-ar',type=str, default="AB")
    parserOpt.add_argument('--conversion', '-co',type=str, default="norm")
    parserOpt.add_argument('--directConversion', '-d',type=bool, default=True)
    parserOpt.add_argument('--windowOperation', '-wo',type=str, default="avg")
    parserOpt.add_argument('--equalizeProteins', '-ep',type=bool, default=False)
    parserOpt.add_argument('--mergeOperation', '-mo',type=str, default='max')
    parserOpt.add_argument('--normalizeProteins', '-np',type=bool, default=False)
    parserOpt.add_argument('--region', '-re',type=str, default=None)
    parserOpt.add_argument('--log', '-l',type=bool, default=True)
    parserOpt.add_argument('--model', '-m',type=str, default="rf")
    parserOpt.add_argument('--regionIndex2', '-r2',type=int, default=None)
    parserOpt.add_argument('--regionIndex1', '-r1',type=int, default=None)
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
    elif args.action == "loadProtein":
        loadProtein(args)
    elif args.action == "loadAllProteins":
        loadAllProteins(args)
    elif args.action == "createWindows":
        createForestDataset(args)
    elif args.action == "createAllWindows":
        createAllWindows(args)
    elif args.action == "plot":
        plotMatrix(args)
    elif args.action == "plotPred":
        plotPredMatrix(args)
    elif args.action == "plotAll":
        plotAllOfDir(args)
    elif args.action == "createArms":
        createAllArms(args)


if __name__ == "__main__":
    main(sys.argv[1:])
    # df = pickle.load(open( SET_D+"4_test.p", "rb" ) )
    # print(df)
    # df = pickle.load(open(SET_D+"1_A_avg.p", "rb" ) )
    # print(df)
