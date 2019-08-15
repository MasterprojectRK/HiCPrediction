#!/usr/bin/env python3

from configurations import *
from tagCreator import createPredictionTag
START =7
END = 23
resultName = "part_"+str(START)+"_"+str(END)+".p"

@predict_options
@click.argument('predictionSets', nargs=-1)
@click.command()
def executePrediction(modelfilepath, basefile,predictionsets, 
                      predictionoutputdirectory):
    checkExtension(modelfilepath, 'z')
    checkExtension(basefile, 'procool')
    model, modelParams = joblib.load(modelfilepath)
    for fileName in predictionsets:
        testSet, setParams = joblib.load(fileName)
        prediction, score = predict(model, testSet, modelParams['conversion'])
        print(score)
        predictionTag = createPredictionTag(modelParams, setParams['chrom'])
        predictionFile =  predictionoutputdirectory + predictionTag + ".precool"
        predictionToMatrix(prediction, basefile,modelParams['conversion'], setParams['chrom'], predictionFile)
        # saveResults(args, prediction, score)




    # c = args.chrom
    # chroms = [item for item in chromStringToList(args.chrom)]
    # modelChroms = [item for item in chromStringToList(args.modelChroms)]
    # print(modelChroms)
    # chromDict = dict()
    # armsDict = dict()
    # setDict = dict()
    # for  x in tqdm(chroms):
        # chromDict[x] = hm.hiCMatrix(CHROM_D+x+".cool")
    # print("Chroms loaded")
    # armDict = divideIntoArms(args, chromDict)
    # print("Arms generated")
    # proteinDict = loadAllProteins(args, armDict)
    # print("\nProteins generated")
    # for  name, arm in tqdm(armDict.items(), desc="Creating set for each arm"):
        # setDict[name] = createDataset(args, name, arm, proteinDict[name])
    # print("\nSets generated")
    # for modelChrom in modelChroms:
        # args.chrom = c
        # args.chroms = modelChrom
        # model = pickle.load(open(tagCreator(args, "model"), "rb" ) ) 
        # combined = createCombinedDataset(setDict)
        # prediction, score = predict(args, model, combined)
        # saveResults(args, prediction, score)
        # predictionPreparation(args, prediction, armDict)

def predict(model, testSet, conversion):
    testSet = testSet.fillna(value=0)
    test_X = testSet[testSet.columns.difference(['first', 'second','chrom','reads'])]
    test_y = testSet['reads']
    test_y = testSet['chrom']
    test_y = test_y.to_frame()
    test_y['standardLog'] = np.log(testSet['reads']+1)
    y_pred = model.predict(test_X)
    test_y['pred'] = y_pred
    y_pred = np.absolute(y_pred)
    test_y['second'] = testSet['second']
    test_y['first'] = testSet['first']
    test_y['distance'] = testSet['distance']
    test_y['predAbs'] = y_pred
    # if args.conversion == "norm":
        # target = 'normTarget'
        # reads = y_pred * maxV
    # elif args.conversion == "log":
        # target = 'logTarget'
        # reads = y_pred * np.log(maxV)
        # reads = np.exp(reads) - 1
    if conversion == 'none':
        target = 'reads'
        reads = y_pred
    elif conversion == 'standardLog':
        target = 'standardLog'
        reads = y_pred
        reads = np.exp(reads) - 1
    test_y['reads'] = testSet['reads']
    test_y['predReads'] = reads
    score = model.score(test_X,test_y[target])
    test_y = test_y.set_index(['first','second'])
    return test_y, score

def predictionToMatrix(pred, baseFilePath,conversion, chromosome, predictionFilePath):
    with h5py.File(baseFilePath, 'a') as baseFile:
        if os.path.isfile(predictionFilePath):
            os.remove(predictionFilePath)
        with h5py.File(predictionFilePath, 'a') as predictionFile:
            baseFile.copy(chromosome+'/pixels', predictionFile, name="pixels/") 
            baseFile.copy(chromosome+'/bins', predictionFile, name="bins/") 
            baseFile.copy(chromosome+'/chroms', predictionFile, name="chroms/") 
            baseFile.copy(chromosome+'/indexes', predictionFile, name="indexes/") 
            if conversion == "standardLog":
                convert = lambda val: np.exp(val) - 1
            elif conversion == "none":
                convert = lambda val: val
            pred['idx1'] = pred.index.codes[0]
            pred['idx2'] = pred.index.codes[1]
            pred['predConv'] = convert(pred['pred'])
            readPath = '/pixels/'
            del predictionFile[readPath+"bin1_id"]
            del predictionFile[readPath+"bin2_id"]
            del predictionFile[readPath+"count"]
            predictionFile[readPath+"bin1_id"] = np.array(pred['predConv'])
            predictionFile[readPath+"bin2_id"] = np.array(pred['idx1'])
            predictionFile[readPath+"count"] = np.array(pred['idx2'])
            


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

if __name__ == '__main__':
    executePrediction()
