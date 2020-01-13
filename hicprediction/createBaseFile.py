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
        msg = "Aborted. The following input files are not narrowPeak or broadPeak files:\n"
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

    outDirectory = internaloutdir
    ### call of function responsible for cutting and storing the chromosomes
    chromosomeDict = addGenome(matrixfile, basefile, chromosomeList,
                               outDirectory)
    with h5py.File(basefile, 'a'):
        ### iterate over possible combinations for protein settings
        for setting in conf.getBaseCombinations():
            params['peakColumn'] = setting['peakColumn']
            params['normalize'] = setting['normalize']
            params['mergeOperation'] = setting['mergeOperation']
            ### get protein files with given paths for each setting
            proteins = getProteinFiles(proteinfiles, params)
            proteinTag =createProteinTag(params)
            ### literate over all chromosomes in list 
            for chromosome in tqdm(chromosomeDict.keys(), desc= 'Iterate chromosomes'):
                ### get bin boundaries of HiC matrix
                cuts = chromosomeDict[chromosome].cut_intervals
                cutsStart = [cut[1] for cut in cuts]
                ### call function that actually bins the proteins
                proteinData = loadProtein(proteins, chromosome,cutsStart,params)
                proteinChromTag = proteinTag + "_" + chromosome
                ### store binned proteins in base file
                store = pd.HDFStore(basefile)
                store.put(proteinChromTag, proteinData)
                store.get_storer(proteinChromTag).attrs.metadata = params
                store.close()
    print("\n")

def getProteinFiles(proteinFiles, params):
    """ function responsible of loading the protein files from the paths that
    were given
    Attributes:
        proteinfiles -- list of paths of the protein files to be processed
        params -- dictionary with parameters
    """
    proteins = dict()
    for path in proteinFiles:
        a = pybedtools.BedTool(path)
        malformedFeatures = [features for features in a if len(features.fields) not in [9,10]]
        if malformedFeatures:
            msg = "protein file {0:s} seems to be an invalid narrow- or broadPeak file\n"
            msg += "there are rows with more than 10 or less than 9 columns"
            sys.exit(msg.format(path))
        ### compute min and max for the normalization
        b = a.to_dataframe()
        c = b.iloc[:,params['peakColumn']]
        minV = min(c)
        maxV = max(c) - minV
        if maxV == 0:
            maxV = 1.0
        items = []
        if params['normalize']:
            ### normalize rows if demanded
            for row in a:
                row[params['peakColumn']] = str((float(\
                    row[params['peakColumn']]) - minV) / maxV)
                items.append(row)
            a = pybedtools.BedTool(items)
        proteins[path] = a
    return proteins

def loadProtein(proteins, chromName, cutsStart, params):
    """
    bin protein data and store for the given chromosome
    Attributes:
        protein -- list of loaded bed files for each protein
        chromName --  index of specific chromosome
        cutsStart -- starting positions of bins
        params -- dictionary with parameters
    """

    i = 0
    allProteins = []
    ### create binning function as demanded
    if params['mergeOperation'] == 'avg':
        merge = np.mean
    elif params['mergeOperation'] == 'max':
        merge = np.max
    ### create data structure
    for cut in cutsStart:
        allProteins.append(np.zeros(len(proteins) + 1))
        allProteins[i][0] = cut
        i += 1
    i = 0
    columns = ['start']
    ### iterate proteins 
    for tup in tqdm(proteins.items(), desc = 'Proteins converting'):
        a = tup[1]
        columns.append(str(i))
        ### filter for specific chromosome and sort
        a = a.filter(chrom_filter, c=str(chromName))
        a = a.sort()
        values = dict()
        for feature in a:
            peak = feature.start
            if len(feature.fields) == 10 and int(feature.fields[9]) != -1: 
                #narrowPeak files with called peaks
                #the peak column (9) is an offset to "feature.start"
                peak += int(feature.fields[9])
            else:
                #narrowPeak files without called peaks
                #and broadPeak files, which have no peak column
                peak += feature.stop
                peak = int(peak / 2)
            ### get bin index of peak
            pos = bisect.bisect_right(cutsStart, peak)
            if pos in values:
                values[pos].append(float(feature[params['peakColumn']]))
            else:
                values[pos] = [float(feature[params['peakColumn']])]
        #j = 0
        for key, val in values.items():
            ### bin proteins for each bin
            score = merge(val)
            allProteins[key - 1][i+1] = score
        i += 1
    data = pd.DataFrame(allProteins,columns=columns, index=range(len(cutsStart)))
    return data

def addGenome(matrixFile, baseFilePath, chromosomeList, outDirectory):
    """
    function that cuts genome HiC matrix into chromosomes and stores them
    internally
    Attributes:
        matrixfile -- path to input HiC matrix
        baseFilePath --  output path for base file
        chromosomeList -- list of chromosomes to be processed
        outDirectory --  output path for chromosomes
    """

    chromosomeDict = {}
    with h5py.File(baseFilePath, 'a') as baseFile:
        for i in tqdm(chromosomeList,desc='Converting chromosomes'):
            filename = os.path.basename(matrixFile)
            outMatrix = os.path.join(outDirectory, filename)
            tag = outMatrix.rstrip(".cool") \
                    +"_chr" + str(i) +".cool"
            chromTag = "chr" + str(i)
            ### create process to cut chromosomes
            sub2 = "hicAdjustMatrix -m "+matrixFile +" --action keep" \
                        +" --chromosomes " + chromTag + " -o " + tag
            ### execute call if necessary
            if not chromTag in baseFile:
                subprocess.call(sub2,shell=True)
                baseFile[chromTag] = tag
            elif not os.path.isfile(baseFile[chromTag][()]):
                subprocess.call(sub2,shell=True)
            ### store HiC matrix
            chromosomeDict[chromTag] =hm.hiCMatrix(tag)
    return chromosomeDict

def chrom_filter(feature, c):
        return feature.chrom == c

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


if __name__ == '__main__':
    loadAllProteins()

