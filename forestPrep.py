from hiCOperations import *

START = 14
END = 23
resultName = "part_"+str(START)+"_"+str(END)+".p"

def predict(args, model = None, setC = None):
    if model == None:
        model = pickle.load(open( tagCreator(args, "model"), "rb" ) )
    if args.conversion == "log" or args.conversion == "norm":
        ma = hm.hiCMatrix(CHROM_D +args.chrom+".cool")
        matrix = ma.matrix.todense()
        maxV = np.max(matrix)
    tmp = args.chroms
    args.chroms = args.chrom
    path =  tagCreator(args, "setC")
    if not os.path.isfile(path):
        createCombinedDataset(args)
    setC = pickle.load(open( tagCreator(args, "setC"), "rb" ) )
    args.chroms = tmp
    setC = setC.fillna(value=0)
    test_X = setC[setC.columns.difference(['first', 'second','chrom','reads', 'logTarget',
                                       'normTarget'])]
    test_y = setC['reads']
    test_y = setC['chrom']
    test_y = test_y.to_frame()
    test_y['normTarget'] = setC['normTarget']
    test_y['logTarget'] = setC['logTarget']
    test_y['standardLog'] = np.log(setC['reads']+1)
    y_pred = model.predict(test_X)
    test_y['pred'] = y_pred
    y_pred = np.absolute(y_pred)
    test_y['second'] = setC['second']
    test_y['first'] = setC['first']
    test_y['distance'] = setC['distance']
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
    elif args.conversion == 'standardLog':
        target = 'standardLog'
        reads = y_pred
        reads = np.exp(reads) - 1
    test_y['reads'] = setC['reads']
    test_y['predReads'] = reads
    score = model.score(test_X,test_y[target])
    print(score)

    test_y = test_y.set_index(['first','second'])
    if args.directConversion == 1:
        predictionPreparation(args, test_y) 
    elif args.directConversion == 2:
        pickle.dump(test_y, open(tagCreator(args, "pred"), "wb" ) )
    elif args.directConversion == 0:

        saveResults(args, test_y, score)

def createCombinedDataset():
    data = pd.DataFrame() 
    for f in chroms:
        data = data.append(df)
    return data

def plotGraphics(args):
    plt.plot(indices, values)
    plt.savefig(tagCreator(args, "plot"))

def saveResults(args, y, score):
    y_pred = y['predReads']
    y_true = y['reads']
    new = y.groupby('distance', group_keys=False)[['reads',
                                                   'predReads']].corr(method='pearson')
    new = new.iloc[0::2,-1]
    values = new.values
    indices = new.index.tolist()
    indices = list(map(lambda x: x[0], indices))
    indices = np.asarray(indices)
    indices = indices /5000
    indices = indices /(len(indices)+2)

    aucScore = auc(indices, values)
    df = pickle.load(open(RESULTPART_D+resultName, "rb" ) )
    cols = [score, r2_score(y_pred,y_true),mean_squared_error(y_pred, y_true),
            mean_absolute_error(y_pred, y_true), mean_squared_log_error(y_pred, y_true),
            aucScore, args.windowOperation, args.mergeOperation,
            args.model, args.equalizeProteins,args.normalizeProteins, args.conversion,
            args.chrom, args.chroms, args.loss , args.estimators]
    print(cols)
    cols.extend(values)
    df.loc[tagCreator(args,"pred").split("/")[-1],:]= pd.Series(cols, index=df.columns )
    df = df.sort_values(by=['trainChroms', 'chrom', 'model','conversion', 'window',
                  'merge', 'ep', 'np'])
    pickle.dump(df, open(RESULTPART_D+resultName, "wb" ) )


def startTraining(args, df):
    if args.model == "rf":
        model = RandomForestRegressor(max_features='sqrt',random_state=5,\
                    n_estimators =args.estimators,n_jobs=4, verbose=3, criterion=args.loss)
    elif args.model ==  "ada":
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'verbose':True,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    elif args.model ==  "mlp":
        model = MLPRegressor(hidden_layer_sizes=(20,10,5),verbose=True,
                             early_stopping=True, max_iter=25)
    df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(value=0)
    X = df[df.columns.difference(['first', 'second','chrom',
                                  'reads'])]
    if args.conversion == 'default':
        y = df['reads']
    elif args.conversion == 'standardLog':
        y = np.log(df['reads']+1)
    model.fit(X, y)
    pickle.dump(model, open(tagCreator(args, "model"), "wb" ) ) 


def predictAll(args):
    mergeAndSave()
    shutil.copy(RESULT_D+"baseResults.p",RESULTPART_D + resultName)
    df = pickle.load(open(RESULT_D+"baseResults.p", "rb" ) )
    # for cs in [14]:
    # tcs.extend([])
    tcs = ["1"]
    # tcs.extend(["2_4", "1_6","1_10_14","6_10_14"])
    for cs in tcs:
        args.chroms = str(cs)
        for w in ["avg"]:
                args.windowOperation = w
                for me in ["avg"]:
                    args.mergeOperation = me
                    for m in ["rf"]:
                        args.model = m
                        # for p in ["log"]:
                        for p in ["default", "standardLog", "log"]:
                            args.conversion = p
                            for n in [False]:
                                args.normalizeProteins = n
                                for e in [False]:
                                    args.equalizeProteins = e
                                    if  os.path.isfile(tagCreator(args,"model")):
                                        model = pickle.load(open( tagCreator(args, "model"), "rb" ) )
                                        for c in [9]:
                                        # for c in range(START,END):
                                            args.chrom = str(c)
                                            print(tagCreator(args,"pred"))
                                            if args.directConversion == 0:
                                                if tagCreator(args,"pred").split("/")[-1] in df.index:
                                                    print("Exists")
                                                else:
                                                    predict(args, model)

                                            elif  args.directConversion == 1: 
                                                exists = True

                                                for suf in ["_A", "_B"]:
                                                    args.chrom = str(c) + suf
                                                    if os.path.isfile(ARM_D +args.chrom+".cool"):
                                                        if not os.path.isfile(tagCreator(args,"pred")):
                                                            exists = False
                                                args.chrom = str(c)
                                                if exists == False:
                                                    print(args.chrom)
                                                    predict(args, model)
                                                else:
                                                    print("Both exist")
                                            else:
                                                predict(args, model)

def checkIfAlMetricsExist(args, key):
    if  not os.path.isfile(tagCreator(args, key)):
        return False
    return True

