from hiCOperations import *
START =7
END = 23
resultName = "part_"+str(START)+"_"+str(END)+".p"

def executePrediction(args):
    c = args.chrom
    chroms = [item for item in chromStringToList(args.chrom)]
    modelChroms = [item for item in chromStringToList(args.modelChroms)]
    print(modelChroms)
    chromDict = dict()
    armsDict = dict()
    setDict = dict()
    for  x in tqdm(chroms):
        chromDict[x] = hm.hiCMatrix(CHROM_D+x+".cool")
    print("Chroms loaded")
    armDict = divideIntoArms(args, chromDict)
    print("Arms generated")
    proteinDict = loadAllProteins(args, armDict)
    print("\nProteins generated")
    for  name, arm in tqdm(armDict.items(), desc="Creating set for each arm"):
        setDict[name] = createDataset(args, name, arm, proteinDict[name])
    print("\nSets generated")
    for modelChrom in modelChroms:
        args.chrom = c
        args.chroms = modelChrom
        model = pickle.load(open(tagCreator(args, "model"), "rb" ) ) 
        combined = createCombinedDataset(setDict)
        prediction, score = predict(args, model, combined)
        saveResults(args, prediction, score)
        predictionPreparation(args, prediction, armDict)


def predictionPreparation(args, pred, armDict):
        print(args.chrom)
        c = args.chrom
        args.chrom = c +"_A"
        predA = pred.loc[pred['chrom'] == args.chrom]
        predictionToMatrix(args, predA, armDict[args.chrom])
        if  c not in ['13','14','15','22']:
            args.chrom = c +"_B"
            predB = pred.loc[pred['chrom'] == args.chrom]
            predictionToMatrix(args, predB, armDict[args.chrom])


def predict(args, model = None, setC = None):
    setC = setC.fillna(value=0)
    test_X = setC[setC.columns.difference(['first', 'second','chrom','reads'])]
    test_y = setC['reads']
    test_y = setC['chrom']
    test_y = test_y.to_frame()
    test_y['standardLog'] = np.log(setC['reads']+1)
    y_pred = model.predict(test_X)
    test_y['pred'] = y_pred
    y_pred = np.absolute(y_pred)
    test_y['second'] = setC['second']
    test_y['first'] = setC['first']
    test_y['distance'] = setC['distance']
    test_y['predAbs'] = y_pred
    # if args.conversion == "norm":
        # target = 'normTarget'
        # reads = y_pred * maxV
    # elif args.conversion == "log":
        # target = 'logTarget'
        # reads = y_pred * np.log(maxV)
        # reads = np.exp(reads) - 1
    if args.conversion == 'default':
        target = 'reads'
        reads = y_pred
    elif args.conversion == 'standardLog':
        target = 'standardLog'
        reads = y_pred
        reads = np.exp(reads) - 1
    test_y['reads'] = setC['reads']
    test_y['predReads'] = reads
    score = model.score(test_X,test_y[target])
    test_y = test_y.set_index(['first','second'])
    return test_y, score

def predictionToMatrix(args, pred, arm):
    mat = arm.matrix.todense()
    factor =  np.max(mat)
    # if args.conversion == "norm":
        # convert = lambda val: val  * factor
    # elif args.conversion == "log":
        # convert = lambda val: np.exp(val  * np.log(factor)) - 1
    if args.conversion == "standardLog":
        convert = lambda val: np.exp(val) - 1
    elif args.conversion == "default":
        convert = lambda val: val
    pred['idx1'] = pred.index.codes[0]
    pred['idx2'] = pred.index.codes[1]
    pred['predConv'] = convert(pred['pred'])
    data = np.array(pred['predConv'])
    row = np.array(pred['idx1'])
    col = np.array(pred['idx2'])
    new = sparse.csr_matrix((data, (row, col)), shape=mat.shape)
    arm.setMatrix(new, arm.cut_intervals)
    arm.save(tagCreator(args, "pred"))


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
    indices = indices / args.binSize
    indices = indices /(len(indices)+2)
    aucScore = auc(indices, values)
    df = pickle.load(open(RESULT_D + "baseResults.p", "rb" ) )
    cols = [score, r2_score(y_pred,y_true),mean_squared_error(y_pred, y_true),
            mean_absolute_error(y_pred, y_true), mean_squared_log_error(y_pred, y_true),
            aucScore, args.windowOperation, args.mergeOperation,
            args.model, args.equalizeProteins,args.normalizeProteins, args.conversion,
            args.chrom, args.chroms, args.loss , args.estimators]
    print(cols)
    cols.extend(values)
    cols.extend([0,args.cellLine, args.modelCellLine,
                 "False",args.binSize])
    df.loc[tagCreator(args,"pred").split("/")[-1],:]= pd.Series(cols, index=df.columns )
    df = df.sort_values(by=['cellLine', 'modelCellLine','trainChroms',\
                            'chrom', 'model','conversion', 'window',\
                  'merge', 'ep', 'np'])
    pickle.dump(df, open(RESULT_D + "baseResults.p", "wb" ) )
