import forestPrep as fp
from hiCOperations import *


def executeTraining(args):
    chroms = [item for item in chromStringToList(args.chroms)]
    print(chroms)
    chromDict = dict()
    armsDict = dict()
    setDict = dict()
    for  x in tqdm(chroms):
        chromDict[x] = hm.hiCMatrix(CHROM_D+x+".cool")
    print("Chroms loaded")
    for  name, chrom in tqdm(chromDict.items()):
        armsDict[name] = divideIntoArms(args, chrom, name) 
    print("Arms generated")
    proteinDict = loadAllProteins(args, armsDict)
    print("Proteins generated")
    for  name, arm in tqdm(armsDict.items()):
        chromName = name.split("_")[0]
        setDict[name] = createDataset(args, name, arm, chromDict[chromName], proteinDict[name])
    print("Sets generated")

def main(args=None):
    args = parseArguments(args)
    if args.action == "train":
        startTraining(args)
    elif args.action == "trainAll":
        trainAll(args)
    elif args.action == "execute":
        executeTraining(args)
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
