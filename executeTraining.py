import forestPrep as fp
from hiCOperations import *
from predict import executePrediction

def executeTraining(args):
    chroms = [item for item in chromStringToList(args.chroms)]
    print(chroms)
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
    combined = createCombinedDataset(setDict)
    print("Starting training")
    fp.startTraining(args, combined)

def main(args=None):
    args = parseArguments(args)
    if args.action == "train":
        startTraining(args)
    elif args.action == "trainAll":
        trainAll(args)
    elif args.action == "execute":
        executeTraining(args)
    elif args.action == "executePrediction":
        executePrediction(args)
    elif args.action == "predict":
        predict(args)
    elif args.action == "predictAll":
        predictAll(args)
    elif args.action == "predToM":
        predictionPreparation(args)
    elif args.action == "storeCM":
        storeMatrixAndCuts(args)
    elif args.action == "trainPredictAll":
        trainAll(args)
        args.directConversion = 0
        predictAll(args)
        # args.directConversion = 1
        # predictAll(args)
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
    elif args.action == "allCombs":
        createAllCombinations(args)
    elif args.action == "plot":
        plotMatrix(args)
    elif args.action == "plotPred":
        plotPredMatrix(args)
    elif args.action == "plotAll":
        plotDir(args)
    elif args.action == "createArms":
        createAllArms(args)
    elif args.action == "mergeAndSave":
        mergeAndSave()


if __name__ == "__main__":
    main(sys.argv[1:])
