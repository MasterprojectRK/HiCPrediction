import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

@click.option('--resultsfiles', '-r', multiple=True, type=click.Path(exists=True, readable=True), help="results file from predict.py in csv format")
@click.option('--legends', '-l', multiple=True, type=str)
@click.option('--outfile', '-o', type=click.Path(writable=True, dir_okay=False, file_okay=True), required=False, default=None, help="path/name of outfile (must end in .png, .svg, .pdf)")
@click.command()
def plotMetrics(resultsfiles, legends, outfile):
    if not resultsfiles:
        return

    if resultsfiles and legends and len(legends) != len(resultsfiles):
        msg = "If specified, number of legends must match number of resultsfiles"
        raise SystemExit(msg)

    if not legends:
        legends = [None for x in resultsfiles]
    
    resultsDfList = []
    for resultfile in resultsfiles:
        try:
            resultsDf = pd.read_csv(resultfile, index_col=False)
            resultsDfList.append(resultsDf)
        except Exception as e:
            msg = str(e) + "\n"
            msg += "could not read results file {:s}, wrong format?"
            msg = msg.format(resultfile)
            print(msg)  

    fig1, ax1 = plt.subplots()
    ax1.set_ylabel("Pearson correlation")
    ax1.set_xlabel("Genomic distance / Mbp")
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,1.2])
    trainChromSet = set()
    targetChromSet = set()
    trainCellLineSet = set()
    targetCellLineSet = set()
    for i, resultsDf in enumerate(resultsDfList):
        try:
            distStratifiedPearsonFirstIndex = resultsDf.columns.get_loc('predictionCellType') + 1 
            resolutionInt = int(resultsDf.loc[0, 'resolution'])
            trainChromSet.add(resultsDf.loc[0, 'modelChromosome'])
            targetChromSet.add(resultsDf.loc[0, 'predictionChromosome'])
            trainCellLineSet.add(resultsDf.loc[0, 'modelCellType'])
            targetCellLineSet.add(resultsDf.loc[0, 'predictionCellType'])
            pearsonXValues = np.array(np.uint32(resultsDf.columns[distStratifiedPearsonFirstIndex:]))
            pearsonXValues = pearsonXValues * resolutionInt / 1000000
            pearsonYValues = np.array(resultsDf.iloc[0, distStratifiedPearsonFirstIndex:])
        except Exception as e:
            msg = str(e) + "\n"
            msg += "results file {:s} does not contain all relevant fields (resolution, distance stratified pearson correlation data etc.)"
            msg = msg.format(resultsfiles[i])
            print(msg)
        
        titleStr = "Pearson correlation vs. genomic distance"
        if len(trainChromSet) == len(targetChromSet) == len(trainCellLineSet) == len(targetCellLineSet) == 1:
            titleStr += "\n {:s}, {:s} on {:s}, {:s}"
            titleStr = titleStr.format(list(trainCellLineSet)[0], list(trainChromSet)[0], list(targetCellLineSet)[0], list(targetChromSet)[0])
        ax1.set_title(titleStr)
        ax1.plot(pearsonXValues, pearsonYValues, label = legends[i])
    
    if not None in legends:
        ax1.legend(frameon=False)
    if not outfile:
        plt.show()
    else:
        if os.path.splitext(outfile)[1] not in ['.png', '.svg', '.pdf']:
            outfile = os.path.splitext(outfile)[0] + '.png'
            msg = "Outfile must have png, pdf or svg file extension.\n"
            msg += "Renamed outfile to {:s}".format(outfile)
            print(msg)
        fig1.savefig(outfile)
    


if __name__ == "__main__":
    plotMetrics()