import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

@click.option('--modelFile', '-mif', required=True, type=click.Path(exists=True), help="model file with feature importances to plot")
@click.option('--outPath', '-o', required=False, type=click.Path(writable=True), default=None, help="path and filename (.png, .pdf, .svg) where result will be written to")
@click.option('--protNames', '-pn', required=False, type=click.Path(exists=True), default=None, help="textfile with protein/histone names")
@click.command()
def plotFeatureImportances(modelfile, outpath, protnames):
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
    
    protNamesDf = None
    try:
        if protnames:
            protNamesDf = pd.read_csv(protnames)
    except Exception as e:
        msg = "protein name file invalid, falling back to generic names"
        print(e, "\n", msg)
        protNamesDf = pd.DataFrame()
    

    try: #old models may not have the 'method' in their params
        #find out number of proteins and name features accordingly
        method = params['method']
        noMiddle = params['noMiddle']
        noStartEnd = params['noStartEnd']
        noDistance = params['noDistance']
    except:
        method = 'generic'
        msg = "could not load method and other params\n" 
        msg += "falling back to generic naming"
        print(msg)   
        
    featNameList = []
    nrOfProteins = None

    if method == 'multiColumn':
        startProtList = []
        endProtList = []
        windowProtList = []
        buildNameList = []
        #compute target number of proteins first
        divisor = 3 #3 features start, end, window for each protein
        if noMiddle:
            divisor -= 1
        if noStartEnd:
            divisor -= 2
        subtraction = 1
        if noDistance:
            subtraction = 0
        nrOfProteins = int((len(indices) - subtraction) / max(divisor,1))
        #now build the feature name list
        if not protNamesDf.empty and protNamesDf.shape[0] == nrOfProteins:
            #feature 0 = prot0 startProt, feature 1 = prot1 start Prot etc.
            startProtList = ["start_" + str(x) for x in protNamesDf['name']]
            endProtList = ["end_" + str(x) for x in protNamesDf['name']]
            windowProtList = ["window_" + str(x) for x in protNamesDf['name']]
        else:    
            startProtList = ["startProt_" + str(x) for x in range(nrOfProteins)]
            endProtList = ["endProt_" + str(x) for x in range(nrOfProteins)]
            windowProtList = ["windowProt_" + str(x) for x in range(nrOfProteins)]
        #build the final name list
        buildNameList = [startProtList, windowProtList, endProtList]
        if noMiddle:
            buildNameList.remove(windowProtList)
        if noStartEnd:
            buildNameList.remove(startProtList)
            buildNameList.remove(endProtList)
        for nameList in buildNameList:
            featNameList.extend(nameList)
        if not noDistance:
            featNameList.append("distance")
            
    elif method == 'oneHot':
        featNameList = ["startProt", "windowProt", "endProt", "distance"]
        #compute target number of proteins first
        nrOfProteins = len(indices) - 4 #distance, start, stop, window
        if noDistance:
            featNameList.remove("distance")
            nrOfProteins += 1
        if noMiddle:
            featNameList.remove("windowProt")
            nrOfProteins += 1
        if noStartEnd:
            featNameList.remove("startProt")
            featNameList.remove("endProt")
            nrOfProteins += 2
        if not protNamesDf.empty and protNamesDf.shape[0] == nrOfProteins:
            featNameList.extend([str(x) for x in protNamesDf['name']])    
        else:
            featNameList.extend(["prot" + str(x) for x in range(nrOfProteins)])     
    else: #generic names
        featNameList = [ str(x) for x in range(len(indices)) ]

    
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
        if not outpath.endswith(".png") and not outpath.endswith(".svg") and not outpath.endswith(".pdf"):
            outpath += ".png"
        importanceFigStr = outpath
    fig1.savefig(importanceFigStr)


if __name__ == "__main__":
    plotFeatureImportances()


    
        
