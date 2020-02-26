import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

@click.option('--resultsfile', '-r', type=click.Path(exists=True, readable=True), help="results file from predict.py in csv format")
@click.option('--outfile', '-o', type=click.Path(writable=True), required=False, default=None, help="path/name of outfile")
@click.command()
def plotMetrics(resultsfile, outfile):
    try:
        resultsDf = pd.read_csv(resultsfile, index_col=False)
    except Exception as e:
        msg = str(e) + "\n"
        msg += "could not read results file, wrong format?"
        raise SystemExit(msg)  

    try:
        distStratifiedPearsonFirstIndex = resultsDf.columns.get_loc('0') 
        resolutionInt = int(resultsDf.loc[0, 'resolution'])
        trainingChromosomeStr = resultsDf.loc[0, 'modelChromosome']
        targetChromsomeStr = resultsDf.loc[0, 'predictionChromosome']
        trainingCellLineStr = resultsDf.loc[0, 'modelCellType']
        targetCellLineStr = resultsDf.loc[0, 'predictionCellType']
        pearsonXValues = np.array(np.uint32(resultsDf.columns[distStratifiedPearsonFirstIndex:]))
        pearsonXValues = pearsonXValues * resolutionInt
        pearsonYValues = np.array(resultsDf.iloc[0, distStratifiedPearsonFirstIndex:])
    except Exception as e:
        msg = str(e) + "\n"
        msg += "results file does not contain resolution or distance stratified pearson correlation data"
        raise SystemExit(msg)

    fig1, ax1 = plt.subplots()
    ax1.plot(pearsonXValues, pearsonYValues)
    titleStr = "Prediction {:s}, {:s} on {:s}, {:s}\n".format(trainingCellLineStr, trainingChromosomeStr, targetCellLineStr, targetChromsomeStr)
    titleStr += "Pearson correlation vs. genomic distance"
    ax1.set_title(titleStr)
    ax1.set_ylabel("Pearson correlation")
    ax1.set_xlabel("Genomic distance")
    ax1.set_ylim([0,1])

    if not outfile:
        plt.show()
    else:
        if not outfile.endswith('.png') \
            and not outfile.endswith('.pdf') \
            and not outfile.endswith('.svg'):
            outfile = os.path.splitext(outfile)[0] + '.png'
            msg = "Outfile must have png, pdf or svg file extension.\n"
            msg += "Renamed outfile to {:s}".format(outfile)
            print(msg)
        fig1.savefig(outfile)
    


if __name__ == "__main__":
    plotMetrics()