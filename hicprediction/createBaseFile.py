#!/usr/bin/env python3
import math
import os
import subprocess
import sys

import click
import cooler.fileops
import h5py
import hicprediction.configurations as conf
import numpy as np
import pandas as pd
import pybedtools
import pyBigWig
from hicprediction.utilities import createProteinTag, initParamDict, checkExtension
from tqdm import tqdm

os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3, suppress=True)

""" Module responsible for the binning of proteins and the cutting of
    the genome HiC matrix into the chromosomes for easier access.
    Creates base file that stores binned proteins and paths to the 
    chromosome HiC matrices.
"""

@conf.protein_options
@click.argument('proteinFiles', nargs=-1)#, help='Pass all of the protein'\
               # +' files you want to use for the prediction. They should be '+\
               # 'defined for the whole genome or at least, for the chromosomes'\
               # + ' you defined in the options. Format must be "narrowPeak, broadPeak or bigwig"')
@click.command()
def loadAllProteins(proteinfiles, basefile, chromosomes,
                   matrixfile,correctmatrix,celltype,resolution,internaloutdir,chromsizefile):
    """
    Main function that is called with the desired path to the base file, a list
    of chromosomes that are to be included, the source HiC file (whole genome),
    the cell type and the resolution of said HiC matrix and most importantly
    the paths to the protein files that should be used
    Attributes:
        proteinfiles -- list of paths of the protein files to be processed
        basefile --  output path for base file
        chromosomes -- list of chromosomes to be processed (without 'chr')
        matrixfile -- path to input HiC matrix
        celltype -- cell line of the input matrix
        resolution -- resolution of the input matrix
        internaloutdir -- where the internally needed per-chromosome matrices are stored
        chromsizefile -- chromosome.sizes file for binning the proteins
    """
    #if matrixfile has been provided, internalOutDir must also be present
    if matrixfile and not internaloutdir:
        msg = "Error: If --matrixFile / -mif is provided, --internalOutDir / -iod cannot be empty\n"
        sys.exit(msg)


    ### checking extensions of files
    if not checkExtension(basefile, '.ph5'):
        basefilename = os.path.splitext(basefile)[0]
        basefile = basefilename + ".ph5"
        msg = "basefile must have .ph5 file extension\n"
        msg += "renamed to {:s}"
        print(msg.format(basefile))
    #remove existing basefiles, since they sometimes caused problems
    if os.path.isfile(basefile):
        os.remove(basefile)
    
    #split up the inputs and separate by file extensions
    protPeakFileList = [fileName for fileName in proteinfiles if checkExtension(fileName,['.narrowPeak', '.broadPeak'])]
    bigwigFileList = [fileName for fileName in proteinfiles if checkExtension(fileName, 'bigwig')]
    wrongFileExtensionList = [fileName for fileName in proteinfiles \
        if not fileName in protPeakFileList and not fileName in bigwigFileList]
    if wrongFileExtensionList:
        msg = "The following input files are neither narrowPeak / broadPeak nor bigwig files and cannot be processed:\n"
        msg += ", ".join(wrongFileExtensionList)
        print(msg)
    if not protPeakFileList and not bigwigFileList:
        msg = "Nothing to process. Exiting"
        print(msg)
        return 

  
    ### creation of parameter set
    params = initParamDict()
    params['resolution'] = resolution
    params['cellType'] = celltype
    ### conversion of desired chromosomes to list
    if chromosomes:
        chromosomeList = chromosomes.split(',')
        chromosomeList = [chrom.strip() for chrom in chromosomeList]
    else:
        chromosomeList = [str(chrom) for chrom in range(1, 23)]

    params['chromSizes'] = getChromSizes(chromosomeList, chromsizefile)
    params['proteinFileNames'] = [os.path.basename(x) for x in protPeakFileList + bigwigFileList]

    ###load protein data from files and store into python objects
    proteinData = getProteinFiles(protPeakFileList + bigwigFileList)
    
    ### iterate over all possible combinations of settings (merging)
    for setting in conf.getBaseCombinations():
        ### get settings and file tag for each combination
        params['mergeOperation'] = setting['mergeOperation']
        proteinTag =createProteinTag(params)
       
        for chromosome in tqdm(params['chromSizes'], desc= 'Iterate chromosomes'):   
            ### get protein data from each object into a dataframe and store in a list
            binnedProteins = []
            for proteinfile in proteinData.keys():
                binnedProteins.append(loadProteinData(proteinData[proteinfile], chromosome, params))
            ### merge the binned protein dataframes from the list into a single dataframe
            for i in range(len(binnedProteins)):
                binnedProteins[i].columns = [str(i)] #rename signalValue columns to make joining easy
            maxBinInt = math.ceil(params['chromSizes'][chromosome] / int(resolution))
            proteinDf = pd.DataFrame(columns=['bin_id'])
            proteinDf['bin_id'] = list(range(0,maxBinInt))
            proteinDf.set_index('bin_id', inplace=True)
            proteinDf = proteinDf.join(binnedProteins, how='outer')
            proteinDf.fillna(0.0,inplace=True)       
            
            ### store binned proteins in base file
            proteinChromTag = proteinTag + "_chr" + chromosome
            store = pd.HDFStore(basefile)
            store.put(proteinChromTag, proteinDf, format='table')
            store.get_storer(proteinChromTag).attrs.metadata = params
            store.close()

    #if a matrixfile has been provided, cut it into chromosomes
    #and store the resulting matrices internally
    #these matrices can later be used for training
    if matrixfile:
        for chromosome in params['chromSizes']:
            cutHicMatrix(matrixfile, chromosome, internaloutdir, basefile)
            if correctmatrix:
                correctHiCMatrix(matrixfile, chromosome, internaloutdir)
    params['matrixCorrection'] = correctmatrix    
    
             

def getProteinFiles(pProteinFileList):
    """ function responsible of loading the protein files from the paths that
    were given
    Attributes:
        pProteinFileList -- list of paths of the protein files to be processed
    returns:
        dict with filenames as keys and python objects as values
    """
    proteinData = dict()
    for path in pProteinFileList:
        if path.endswith('Peak'):
            proteinData[path] = getDataFromPeakFile(path)
        if path.endswith('.bigwig'):
            proteinData[path] = getDataFromBigwigFile(path)
    return proteinData

def getDataFromPeakFile(pPeakFilePath):
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
    protData = bedToolFile.to_dataframe()
    columnNames = ['chrom', 'chromStart', 'chromEnd', 'name', 'score',
                            'strand', 'signalValue', 'pValue', 'qValue', 'peak']
    if protData.shape[1] == 10: #narrowPeak format
        protData.columns = columnNames
        mask = protData['peak'] == -1 
        if mask.any(): #no peak summit called
            protData[mask]['peak'] = ( (protData[mask]['chromEnd']-protData[mask]['chromStart'])/2 ).astype('uint32')
    elif protData.shape[1] == 9: #broadPeak format, generally without peak column in the data
        protData.columns = columnNames[0:9]
        protData['peak'] = ((protData['chromEnd'] - protData['chromStart']) / 2).astype('uint32')
    return protData

def getDataFromBigwigFile(pBigwigFilePath):
    try:
        bigwigFile = pyBigWig.open(pBigwigFilePath)
    except:
        msg = "bigwig file {0:s} could not be parsed"
        raise ValueError(msg.format(pBigwigFilePath))
    if not bigwigFile.isBigWig():
        msg = "bigwig file {0:s} is not a proper bigwig file"
        raise ValueError(msg.format(pBigwigFilePath))
    return bigwigFile

def loadProteinData(pProteinDataObject, pChrom, pParams):
    """
    bin protein data and store for the given chromosome
    Attributes:
        protein -- list of loaded bed files for each protein
        chromName --  index of specific chromosome
        cutsStart -- starting positions of bins
        params -- dictionary with parameters
    """
    if isinstance(pProteinDataObject, pyBigWig.pyBigWig):
        dataframe = loadProteinDataFromBigwig(pProteinDataObject, pChrom, pParams)
    else:
        dataframe = loadProteinDataFromPeaks(pProteinDataObject, pChrom, pParams)
    return dataframe
    
def loadProteinDataFromBigwig(pProteinDataObject, pChrom, pParams):
    #pProteinDataObject is instance of class pyBigWig.pyBigWig
    chrom = "chr" + str(pChrom)
    resolution = int(pParams['resolution'])
    #compute signal values (stats) over resolution-sized bins
    chromsize = pProteinDataObject.chroms(chrom)
    chromStartList = list(range(0,chromsize,resolution))
    chromEndList = list(range(resolution,chromsize,resolution))
    chromEndList.append(chromsize)
    if pParams['mergeOperation'] == 'max':
        mergeType = 'max'
    else:
        mergeType = 'mean'
    signalValueList = pProteinDataObject.stats(chrom, 0, chromsize, nBins=len(chromStartList), type=mergeType)
    proteinDf = pd.DataFrame(columns = ['bin_id', 'chromStart', 'chromEnd', 'signalValue'])
    proteinDf['chromStart'] = chromStartList
    proteinDf['chromEnd'] = chromEndList
    proteinDf['signalValue'] = signalValueList 
    #compute bin ids
    proteinDf['bin_id'] = (proteinDf['chromStart'] / resolution).astype('uint32')
    #drop not required columns and switch index => same format as for peak file
    proteinDf.drop(columns=['chromStart', 'chromEnd'], inplace=True)
    proteinDf.set_index('bin_id',drop=True, inplace=True)
    return proteinDf

def loadProteinDataFromPeaks(pProteinDataObject, pChrom, pParams):    
    #pProteinDataObject is a pandas dataframe
    mask = pProteinDataObject['chrom'] == "chr" + str(pChrom)
    proteinDf = pProteinDataObject[mask].copy()
    proteinDf.drop(columns=['chrom','name', 'score', 'strand', 'pValue', 'qValue'], inplace=True)
    resolution = int(pParams['resolution'])
    proteinDf['bin_id'] = ((proteinDf['chromStart'] + proteinDf['peak'])/resolution).astype('uint32')
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


def cutHicMatrix(pMatrixFile, pChrom, pOutDir, pBasefile):
    #cut HiC matrices for single chrom and store in outDir
    if not cooler.fileops.is_cooler(pMatrixFile) and not pMatrixFile.endswith(".cool"):
        msg = "Warning: {:s} is no cooler file and therefore ignored"
        msg = msg.format(pMatrixFile)
        print(msg)
        return
    
    chromTag = "chr" + str(pChrom)
    inFileName = os.path.basename(pMatrixFile)
    outFileName = os.path.splitext(inFileName)[0] + "_" + chromTag + ".cool"
    outMatrixFileName = os.path.join(pOutDir, outFileName)
    
    ### create process to cut chromosomes
    msg = "trying to store Hi-C matrix for chrom {0:s} as {1:s}"
    print(msg.format(chromTag, outMatrixFileName))
    storedSuccessfully = False
    try:
        hicAdjustMatrixProcess = "hicAdjustMatrix -m "+ pMatrixFile + " --action keep" \
                            +" --chromosomes " + chromTag + " -o " + outMatrixFileName
        subprocess.check_call(hicAdjustMatrixProcess, shell=True)
        storedSuccessfully = True
    except Exception as e:
        msg = "Adjusting and storing orignal matrix internally failed\n"
        msg += "Original error message: " + str(e) + "\n"
        msg += "Retrying with different chrom naming scheme\n"
        print(msg)
    if not storedSuccessfully:
        try:
            hicAdjustMatrixProcess = "hicAdjustMatrix -m "+ pMatrixFile + " --action keep" \
                            +" --chromosomes " + pChrom + " -o " + outMatrixFileName
            subprocess.check_call(hicAdjustMatrixProcess, shell=True)
            storedSuccessfully = True
            
        except Exception as e:
            msg = str(e) + "\n"
            msg += "Adjusting and storing the provided matrix failed terminally. Check chromosome names."
            raise SystemExit(msg)
    if storedSuccessfully:
        print("successfully stored matrix internally")
        with h5py.File(pBasefile, 'a') as baseFile:
            baseFile[chromTag] = outMatrixFileName 

def correctHiCMatrix(pMatrixFile, pChrom, pOutDir):
    chromTag = "chr" + str(pChrom)
    inFileName = os.path.basename(pMatrixFile)
    outFileName = os.path.splitext(inFileName)[0] + "_" + chromTag + ".cool"
    outMatrixFileName = os.path.join(pOutDir, outFileName)

    msg = "Correcting HiC Matrix using Knight-Ruiz method"
    print(msg)
    storedSuccessfully = False
    try:
        hicBalanceMatrixProcess = "hicCorrectMatrix correct -m " + outMatrixFileName \
                             + " --correctionMethod KR" + " --chromosomes " + chromTag \
                             + " --verbose -o " + outMatrixFileName
        subprocess.check_call(hicBalanceMatrixProcess, shell=True)
        storedSuccessfully = True
    except Exception as e:
        msg = "KR-correction failed initially, retrying with different chrom naming scheme\n"
        msg += " Original error message: " + str(e) + "\n"
        print(msg)
    if not storedSuccessfully:
        try:
            hicBalanceMatrixProcess = "hicCorrectMatrix correct -m " + outMatrixFileName \
                                + " --correctionMethod KR" + " --chromosomes " + pChrom \
                                + " --verbose -o " + outMatrixFileName
            subprocess.check_call(hicBalanceMatrixProcess, shell=True)
            storedSuccessfully = True
        except Exception as e:
            msg = str(e) + "\n"
            msg += "KR-correction failed again. Check cooler file."
            return
    if storedSuccessfully: 
        msg = "KR-correction successful"
        print(msg)


if __name__ == '__main__':
    loadAllProteins() # pylint: disable=no-value-for-parameter
