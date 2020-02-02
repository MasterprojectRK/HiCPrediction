import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

@click.option('--modelFile', '-mif', required=True, type=click.Path(exists=True), help="model file with feature importances to plot")
@click.option('--outPath', '-o', required=False, type=click.Path(writable=True), default=None, help="path and filename (.png) where result will be written to")
@click.command()
def plotFeatureImportances(modelfile, outpath):
    try:
        model, params = joblib.load(modelfile)
    except Exception as e:
        msg = str(e) + "\n"
        msg += "could not load model and params from input file, maybe wrong format?"
        raise ValueError(msg)

    try:
        importances = model.feature_importances_
    except Exception as e:
        msg = str(e) + "\n"
        msg += "model does not have feature importances, maybe wrong format?"
        raise ValueError(msg)

    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    
    #find out number of proteins and name features accordingly
    featNameList = []
    nrOfProteins = None
    try: #not all models have the 'method' in their params
        if params['method'] == "multiColumn":
            #compute number of proteins first
            divisor = 3 #3 features start, end, window for each protein
            if params['noMiddle'] == True:
                divisor -= 1
            if params['noStartEnd'] == True:
                divisor -= 2
            subtraction = 1
            if params['noDistance'] == True:
                subtraction = 0
            nrOfProteins = int((len(indices) - subtraction) / max(divisor,1))
            #now build the feature name list
            #feature 0 = prot0 startProt, feature 1 = prot1 start Prot etc.
            startProtList = ["startProt" + str(x) for x in range(nrOfProteins)]
            endProtList = ["endProt" + str(x) for x in range(nrOfProteins)]
            windowProtList = ["windowProt" + str(x) for x in range(nrOfProteins)]
            buildNameList = [startProtList, windowProtList, endProtList]
            if params['noMiddle'] == True:
                buildNameList.remove(windowProtList)
            if params['noStartEnd'] == True:
                buildNameList.remove(startProtList)
                buildNameList.remove(endProtList)
            for nameList in buildNameList:
                featNameList.extend(nameList)
            if not params['noDistance'] == True:
                featNameList.append("distance")
            
        if params['method'] == 'oneHot':
            nrOfProteins = len(indices) - 4 #distance, start, stop, window
            featNameList = ["startProt", "windowProt", "endProt", "distance"]
            if params['noDistance'] == True:
                nrOfProteins += 1
                featNameList.remove("distance")
            if params['noMiddle'] == True:
                nrOfProteins += 1
                featNameList.remove("windowProt")
            if params['noStartEnd'] == True:
                nrOfProteins += 2
                featNameList.remove("startProt")
                featNameList.remove("endProt")
            featNameList.extend(["prot" + str(x) for x in range(nrOfProteins)])
    except:
        print("Could not load required parameters for naming features. Falling back to enumeration")
        featNameList = [str(x) for x in range(len(indices))] 
    
    print()
    print("Feature importances:")
    for f in range(len(featNameList)):
        print("{0:d}. feature {1:d} ({2:s}; {3:.2f})".format(f + 1, indices[f], featNameList[indices[f]], importances[indices[f]]))
    print()

    nrTrees = len(model.estimators_)
    imgWidth = max(5, np.round(len(indices)/7))
    imgHeight = 0.75*imgWidth
    fig1, ax1 = plt.subplots(constrained_layout=True, figsize=(imgWidth, imgHeight))
    ax1.set_title("Feature importances and std deviations ({:d} trees)".format(nrTrees))
    ax1.bar(2*indices, importances[indices],
            color="r", yerr=std[indices], align="center", width=1.5)
    ax1.set_xticks(2*indices)
    ax1.set_xticklabels(np.array(featNameList)[indices], rotation=90, fontsize=6)
    ax1.set_ylim([0.0,1.0]) #allow comparing results from different params, datasets etc.
    ax1.set_yticks(np.linspace(0,1,11))
    ax1.set_xlabel("feature name")
    ax1.set_ylabel("relative feature importance")
    importanceFigStr = os.path.splitext(modelfile)[0] + "_importanceGraph.png"
    if outpath:
        if not outpath.endswith(".png"):
            outpath += ".png"
        importanceFigStr = outpath
    fig1.savefig(importanceFigStr)


if __name__ == "__main__":
    plotFeatureImportances()


    
        
