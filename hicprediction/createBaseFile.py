#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import sys
import hicprediction.configurations as conf
import click
import h5py
from tqdm import tqdm
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import pybedtools
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3, suppress=True)
import bisect
from hicprediction.tagCreator import createProteinTag
from hicmatrix import HiCMatrix as hm
import subprocess
import pyBigWig

""" Module responsible for the binning of proteins and the cutting of
    the genome HiC matrix into the chromosomes for easier access.
    Creates base file that stores binned proteins and paths to the 
    chromosome HiC matrices.
"""

@conf.protein_options
@click.argument('proteinFiles', nargs=-1)#, help='Pass all of the protein'\
               # +' files you want to use for the prediction. They should be '+\
               # 'defined for the whole genome or at least, for the chromosomes'\
               # + ' you defined in thee options. Format must be "narrowPeak"')
@click.command()
def loadAllProteins(proteinfiles, basefile, chromosomes,
                   matrixfile,celltype,resolution,internaloutdir,chromsizefile):
    """
    Main function that is called with the desired path to the base file, a list
    of chromosomes that are to be included, the source HiC file (whole genome),
    the cell type and the resolution of said HiC matrix and most importantly
    the paths to the protein files that should be used
    Attributes:
        proteinfiles -- list of paths of the protein files to be processed
        basefile --  output path for base file
        chromosomes -- list of chromosomes to be processed
        matrixfile -- path to input HiC matrix
        celltype -- cell line of the input matrix
        resolution -- resolution of the input matrix
        outdir -- where the internally needed per-chromosome matrices are stored
    """
    ### checking extensions of files
    if not conf.checkExtension(basefile, '.ph5'):
        basefilename = os.path.splitext(basefile)[0]
        basefile = basefilename + ".ph5"
        msg = "basefile must have .ph5 file extension\n"
        msg += "renamed to {0:s}"
        print(msg.format(basefile))
    protPeakFileList = [fileName for fileName in proteinfiles if conf.checkExtension(fileName,'.narrowPeak', '.broadPeak')]
    bigwigFileList = [fileName for fileName in proteinfiles if conf.checkExtension(fileName, 'bigwig')]
    wrongFileExtensionList = [fileName for fileName in proteinfiles \
        if not fileName in protPeakFileList and not fileName in bigwigFileList]
    if wrongFileExtensionList:
        msg = "Aborted. The following input files are neither narrowPeak / broadPeak nor bigwig files:\n"
        msg += ", ".join(wrongFileExtensionList)
        sys.exit(msg)
    ### creation of parameter set
    params = dict()
    params['resolution'] = resolution
    params['cellType'] = celltype
    ### conversion of desired chromosomes to list
    if chromosomes:
        chromosomeList = chromosomes.split(',')
    else:
        chromosomeList = range(1, 23)
    #outDirectory = resource_filename('hicprediction',
                                               #'InternalStorage') +"/"
    params['chromList'] = chromosomeList
    params['chromSizes'] = getChromSizes(chromosomeList, chromsizefile)

    with h5py.File(basefile, 'a'):
        ### iterate over possible combinations for protein settings
        for setting in conf.getBaseCombinations():
            params['peakColumn'] = 6 #column with signal value in narrowPeak and broadPeak files
            params['normalize'] = setting['normalize']
            params['mergeOperation'] = setting['mergeOperation']
            ### get protein files with given paths for each setting
            proteinTag =createProteinTag(params)
            ###load protein data from files 
            proteinData = getProteinFiles(proteinfiles, params)
            ### literate over all chromosomes in list
            for chromosome in tqdm(params['chromList'], desc= 'Iterate chromosomes'):
                ### get bins for the proteins
                bins = getBins(params['chromSizes'][chromosome], params['resolution'])
                binnedProteins = []
                for proteinfile in proteinData.keys():
                    binnedProteins.append(loadProteinData(proteinData[proteinfile], chromosome, bins, params))
                proteinChromTag = proteinTag + "_" + chromosome
                ### store binned proteins in base file
                store = pd.HDFStore(basefile)
                store.put(proteinChromTag, binnedProteins)
                store.get_storer(proteinChromTag).attrs.metadata = params
                store.close()
    print("\n")

def getProteinFiles(pProteinFileList, pParams):
    """ function responsible of loading the protein files from the paths that
    were given
    Attributes:
        proteinfiles -- list of paths of the protein files to be processed
        params -- dictionary with parameters
    """
    proteinData = dict()
    for path in pProteinFileList:
        if path.endswith('Peak'):
            proteinData[path] = getProteinDataFromPeakFile(path, pParams)
        if path.endswith('.bigwig'):
            proteinData[path] = getProteinDataFromBigwigFile(path)
    return proteinData

def getProteinDataFromPeakFile(pPeakFilePath, pParams):
    try:        
        bedToolFile = pybedtools.BedTool(pPeakFilePath)
        malformedFeatures = [features for features in bedToolFile if len(features.fields) not in [9,10]]
    except:
        msg = "could not parse protein peak file {0:s} \n"
        msg += "probably no valid narrowPeak or broadPeak file"
        raise ValueError(msg.format(pPeakFilePath))
    if malformedFeatures:
            msg = "protein file {0:s} seems to be an invalid narrow- or broadPeak file\n"
            msg += "there are rows with more than 10 or less than 9 columns"
            sys.exit(msg.format(pPeakFilePath))
    ### compute min and max for the normalization
    protData = bedToolFile.to_dataframe()
    columnNames = ['chrom', 'chromStart', 'chromEnd', 'name', 'score',
                            'strand', 'signalValue', 'pValue', 'qValue', 'peak']
    if protData.shape[1] == 10: #narrowPeak format
        protData.columns = columnNames
        mask = protData['peak'] == -1 
        if mask.any(): #no peak summit called
            protData.loc[protData[mask]]['peak'] = ( (protData[mask]['chromEnd']-protData[mask]['chromStart'])/2 ).astype('uint32')
    elif protData.shape[1] == 9: #broadPeak format, generally without peak summit
        protData.columns = columnNames[0:8]
        protData['peak'] = ((protData['chromEnd'] - protData['chromStart']) / 2).astype('uint32')
    return protData

def getProteinDataFromBigwigFile(pBigwigFilePath):
    try:
        bigwigFile = pyBigWig.open(pBigwigFilePath)
    except:
        msg = "bigwig file {0:s} could not be parsed"
        raise ValueError(msg.format(pBigwigFilePath))
    if not bigwigFile.isBigWig():
        msg = "bigwig file {0:s} is not a proper bigwig file"
        raise ValueError(msg.format(pBigwigFilePath))
    return bigwigFile

def loadProteinData(pProteinDataObject, pChrom, pBins, pParams):
    """
    bin protein data and store for the given chromosome
    Attributes:
        protein -- list of loaded bed files for each protein
        chromName --  index of specific chromosome
        cutsStart -- starting positions of bins
        params -- dictionary with parameters
    """
    if isinstance(pProteinDataObject, pyBigWig.pyBigWig):
        dataframe = loadProteinDataFromBigwig(pProteinDataObject, pChrom, pBins, pParams)
    else:
        dataframe = loadProteinDataFromPeaks(pProteinDataObject, pChrom, pBins, pParams)
    return dataframe
    
def loadProteinDataFromBigwig(pProteinDataObject, pChrom, pBins, pParams):
    chrom = "chr" + str(pChrom)
    resolution = int(pParams['resolution'])
    tupList = pProteinDataObject.intervals(chrom)
    chromStartList = [x[0] for x in tupList]
    chromEndList = [x[1] for x in tupList]
    signalValueList = [x[2] for x in tupList]
    proteinDf = pd.DataFrame(columns = ['bin_id', 'chromStart', 'chromEnd', 'signalValue'])
    proteinDf['chromStart'] = chromStartList
    proteinDf['chromEnd'] = chromEndList
    proteinDf['signalValue'] = signalValueList
    #filter away zeros...TODO
    proteinDf['bin_id'] = (proteinDf['chromEnd'] / resolution).astype('uint32')
    print(proteinDf.head(10))
    if pParams['mergeOperation'] == 'max':
        binnedDf = proteinDf.groupby('bin_id')[['signalValue']].max()
    else:
        binnedDf = proteinDf.groupby('bin_id')[['signalValue']].mean()
    print(binnedDf.head(10))
    return binnedDf

def loadProteinDataFromPeaks(pProteinDataObject, pChrom, pBins, pParams):    
    #pProteinDataObject is a pandas dataframe
    mask = pProteinDataObject['chrom'] == "chr" + str(pChrom)
    proteinDf = pProteinDataObject[mask].copy()
    proteinDf.drop(columns=['chrom','name', 'score', 'strand', 'pValue', 'qValue'], inplace=True)
    resolution = int(pParams['resolution'])
    proteinDf['bin_id'] = ((proteinDf['chromStart'] + proteinDf['peak'])/resolution).astype('uint32')
    #print(proteinDf.head(10))
    if pParams['mergeOperation'] == 'max':
        binnedDf = proteinDf.groupby('bin_id')[['signalValue']].max()
    else:
        binnedDf = proteinDf.groupby('bin_id')[['signalValue']].mean()
    return binnedDf


def getChromSizes(pChromNameList, pChromSizeFile):
    chromSizeDict = dict()
    try:
        chromSizeDf = pd.read_csv(pChromSizeFile,names=['chrom', 'size'],header=None,sep='\t')
    except:
        msg = "Error: could not parse chrom.sizes file {0:s}\n".format(pChromSizeFile)
        msg += "Maybe wrong format, not tab-separated etc.?"
        raise Exception(msg)
    for chromName in pChromNameList:
        sizeMask = chromSizeDf['chrom'] == "chr" + str(chromName)
        if not sizeMask.any():
            msg = "no entry for chromosome {0:s} in chrom.sizes file".format(chromName)
            raise ValueError(msg)
        elif chromSizeDf[sizeMask].shape[0] > 1:
            msg = "multiple entries for chromosome {0:s} in chrom.sizes file".format(chromName)
            raise ValueError(msg)
        else:
            try:
                chromSizeDict[chromName] = int(chromSizeDf[sizeMask]['size'])
            except ValueError:
                msg = "entry for chromosome {0:s} in chrom.sizes is not an integer".format(chromName)
    return chromSizeDict

def getBins(pChromSize, pResolution):
    resolution = int(pResolution)
    binStartList = list(range(0,pChromSize,resolution))
    binEndList = list(range(resolution,pChromSize,resolution))
    binEndList.append(pChromSize)
    if not len(binStartList) == len(binEndList):
        msg = "bug in getBins"
        sys.exit(msg)
    else:
        return (binStartList, binEndList)

if __name__ == '__main__':
    loadAllProteins()

