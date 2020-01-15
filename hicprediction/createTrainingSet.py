#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import click
import hicprediction.configurations as conf
from hicprediction.tagCreator import createSetTag, createProteinTag
from pkg_resources import resource_filename
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import h5py
import joblib
from hicmatrix import HiCMatrix as hm
import itertools

"""
Module responsible for creating the training sets
Is given an optional path to a centromere file, boolean parameters for
normalization,  and the elimination of centromeres. Furthermore
parameter that determine the general bin operation and the bin operation for
the window proteins. The maximum genomic distance can be set as well as an
output directory for the training sets and the chromosomes that should be
used for the training sets. The base file from the former script
(createBaseFile) must also be passed along
"""

@conf.set_options
@click.command()
def createTrainingSet(chromosomes, datasetoutputdirectory,basefile,\
                   centromeresfile,ignorecentromeres,normalize,
                   internalindir, windowoperation, mergeoperation, 
                   windowsize, cutoutlength, smooth, method):
    """
    Wrapper function
    calls function and can be called by click
    """
    createTrainSet(chromosomes, datasetoutputdirectory,basefile,\
                   centromeresfile,ignorecentromeres,normalize,
                   internalindir, windowoperation, mergeoperation, 
                   windowsize, cutoutlength, smooth, method)

def createTrainSet(chromosomes, datasetoutputdirectory,basefile,\
                   centromeresfile,pIgnoreCentromeres,pNormalize,
                   internalInDir, pWindowOperation, pMergeOperation, 
                   pWindowsize, pCutoutLength, pSmooth, pMethod):
    """
    Main function
    creates the training sets and stores them into the given directory
    Attributes:
            chromosomes -- list of chromosomes to be processed
            datasetoutputdirectory --  directory to store the created sets
            basefile -- file path to base file created in first script
            centromeresfile --  file path  with the positions of the centromeres
            ignorecentromeres --  Boolean to decide if centromeres are cut out
            normalize -- Boolean to decide if proteins are normalized
            internalInDir -- path to directory where the per-chromosome cooler matrices are stored
            windowoperation -- bin operation for windows
            mergeoperation --  bin operations for protein binning
            windowsize --  maximal genomic distance
            peakcolumn -- Column in bed file that contains peak values
    """
    ### check extensions
    if not centromeresfile:
        centromeresfile = resource_filename('hicprediction',\
                'InternalStorage') +"/centromeres.txt"

    if not conf.checkExtension(basefile, '.ph5'):
        msg = "Aborted. Basefile {0:s} has the wrong format (wrong file extension) \n"
        msg += "please specify a .ph5 file"
        sys.exit(msg.format(basefile))

    ### convert chromosomes to list
    if chromosomes:
        chromosomeList = chromosomes.split(',')
    else:
        chromosomeList = range(1, 23)

    ### Iterate over chromosomes
    for chromosome in tqdm(chromosomeList, desc="Iterating chromosomes"):
        chromosome = int(chromosome)
        chromTag = "chr" +str(chromosome)
        ### check if chromosome is along the 22 first chromosomes
        if not chromosome in range(1,23):
            msg = 'Chromosome {0:d} is not in the range of 1 to 22.'
            msg += 'Please provide correct chromosomes'
            sys.exit( msg.format(chromosome) )
        ### create parameter set
        params = dict()
        params['chrom'] = chromTag
        params['windowOperation'] = pWindowOperation
        params['mergeOperation'] = pMergeOperation
        params['normalize'] = pNormalize
        params['ignoreCentromeres'] = pIgnoreCentromeres
        params['windowSize'] = pWindowsize
        proteinTag = createProteinTag(params)
        proteinChromTag = proteinTag + "_" + chromTag
        ### retrieve proteins and parameters from base file
        with pd.HDFStore(basefile) as store:
            proteins = store[proteinChromTag]
            params2 = store.get_storer(proteinChromTag).attrs.metadata
        print(proteins.head())
        ###smoothen the proteins by gaussian filtering, if desired
        if pSmooth > 0.0:
            proteins = smoothenProteins(proteins, pSmooth)
        ### join parameters
        params = {**params, **params2}
        setTag = createSetTag(params) + ".z"
        datasetFileName = os.path.join(datasetoutputdirectory, setTag)
        
        ### try to load HiC matrix
        hiCMatrix = None
        reads = None
        try:
            with h5py.File(basefile, 'r') as baseFile:
                matrixfile = baseFile[chromTag][()]
        except:
            #no matrix was present when creating the basefile 
            #which is normal for test set basefiles
            matrixfile = None 
            
        if matrixfile:
            if internalInDir:
                filename = os.path.basename(matrixfile)
                matrixfile = os.path.join(internalInDir, filename)
            if os.path.isfile(matrixfile):
                hiCMatrix = hm.hiCMatrix(matrixfile)
            else:
                msg = ("cooler file {0:s} is missing.\n" \
                          + "Use --iif option to provide the directory where the internal matrices " \
                          +  "were stored when creating the basefile").format(matrixfile)
                sys.exit(msg)  
            ### load reads and bins
            reads = hiCMatrix.matrix
        
        ### pick the correct variant of the dataset creation function
        if pMethod == 'oneHot':
            buildDataset = createDataset2
        else:
            buildDataset = createDataset
        params['method'] = pMethod

        ### if user decided to cut out centromeres and if the chromosome
        ### has one, create datasets for both chromatids and join them
        if pIgnoreCentromeres:
            resolution = int(params['resolution'])
            if pCutoutLength < resolution:
                msg="Error: Cutout length must not be smaller than resolution. Aborting"
                sys.exit(msg)
            elif pCutoutLength < 5*resolution:
                msg = "Warning: cutout length is smaller than 5x resolution\n"
                msg += "Many regions might be cut out."
                print(msg)
            threshold = int(pCutoutLength / resolution)
            starts, ends = findValidProteinRegions(proteins, threshold)
            if not starts or not ends or len(starts) < 1 or len(ends) < 1:
                msg = "No valid protein peaks found. Aborting."
                sys.exit(msg)
            else:
                dfList = []
                for s, e in zip(starts, ends):
                    dfList.append(buildDataset(proteins, reads, pWindowOperation, pWindowsize,
                                chromosome, pStart = s, pEnd = e))
                df = pd.concat(dfList, ignore_index=True, sort=False)     
        else:
            df = buildDataset(proteins, reads, pWindowOperation, pWindowsize,
                               chromosome, pStart=0, pEnd=proteins.shape[0]-1)
        
        if df.empty:
            msg = "Could not create dataset. Aborting"
            sys.exit(msg)

        ### add average contact read stratified by distance to dataset
        for i in tqdm(range(int(pWindowsize)),desc='Adding average read values'):
            df.loc[df['distance'] == i,'avgRead'] =  df[df['distance'] == i]['reads'].mean()
            
        #one-hot encoding for the proteins / protein numbers
        if pMethod == 'oneHot':
            df['proteinNr'] = df['proteinNr'].astype('category')
            df = pd.get_dummies(df, prefix='prot')
        #print(df.head(10))
        #print(df.tail(10))
            
        joblib.dump((df, params), datasetFileName,compress=True ) 


def getCentromerePositions(centromeresfilepath, chromTag, cuts):
    """
    function that loads the centromere file and gets the centromere start and
    end position for the specific chromosome
    Attributes:
        centromeresfilepath -- filepath to centromeresfile
        chromTag --  tag for the specific chromosome
        cuts -- list of bin positions
    """
    f=open(centromeresfilepath, "r")
    fl =f.readlines()
    elems = None
    for x in fl:
        elems = x.split("\t")
        if elems[1] == chromTag:
            break
    start = int(elems[2])
    end = int(elems[3])
    toStart = cuts[cuts < start]
    toEnd = cuts[cuts < end]
    return  len(toStart), len(toEnd)

def createDataset(pProteins, pFullReads, pWindowOperation, pWindowSize,
                   pChrom, pStart, pEnd):
    """
    function that creates the actual dataset for a specific
    chromosome/chromatid
    Attributes:
            proteins -- array with protein peaks
            fullReads --  array with read values
            windowOperation --  window bin operation
            windowsize -- maximal genomic distance
            chrom -- specific chromosome index
            start -- start position of interval to be processed
            end -- end  position of interval to be processed
    """
    df = pd.DataFrame()
    if pEnd > pStart and pEnd >= 0 and pStart >= 0: #end not >= start (greater equal), since we look at diagonals and need two non-equal values
        proteins = pProteins[pStart:pEnd].reset_index(drop=False)
        numberOfProteins = proteins.shape[1] - 1
    
        numberOfDiagonals = min(pEnd-pStart,pWindowSize) #range might be smaller than window size, if centromere close to start / end of chromosome or if valid region small
        trapezIndices = np.mask_indices(proteins.shape[0],maskFunc,k=numberOfDiagonals)
        
        reads = None
        readMatrix = None
        if pFullReads != None:
            readMatrix = pFullReads[pStart:pEnd,pStart:pEnd]
            reads = np.array(readMatrix[trapezIndices])[0] # get only the relevant reads
    
        ### create data frame and fill with values and indices
        strList = [str(x) for x in range(3*(numberOfProteins))]
        cols = ['first', 'second','chrom'] + strList+ ['distance','reads']
        df = pd.DataFrame(columns=cols)
        df['first'] = np.uint32(trapezIndices[0])
        df['second'] = np.uint32(trapezIndices[1])
        df['distance'] = df['second'] - df['first']
        df['chrom'] = np.uint8(pChrom)
        if pFullReads != None:
            df['reads'] = np.float32(reads)
            df['reads'].fillna(0, inplace=True)
    ### iterate over all the proteins and fill the data frame
        for protein in tqdm(range(numberOfProteins), desc="Converting Proteins to dataset"):
            protIndex = str(protein)
            #start proteins
            df[protIndex] = list(proteins[protIndex][df['first']])
            #end proteins
            df[str(numberOfProteins * 2 + protein)] = list(proteins[protIndex][df['second']])
        
            #window proteins
            winSize = min(pEnd-pStart, pWindowSize)
            windowDf = buildWindowDataset(proteins, protein, winSize, pWindowOperation)
            #get the window proteins into an array and slice it to get all values at once 
            #there might be a more efficient way using pandas
            distWindowArr = windowDf.to_numpy()
            slice1 = list(df['second'])
            slice2 = list(df['distance'])
            slice3 = (slice1, slice2)
            windowProteins = np.array(distWindowArr[slice3])
            df[str(numberOfProteins + protein)] = np.float32(windowProteins)

        #consider offset
        df['first'] += pStart
        df['second'] += pStart
    return df

def createDataset2(pProteins, pFullReads, pWindowOperation, pWindowSize,
                   pChrom, pStart, pEnd):
    
    df = pd.DataFrame()
    
    if pEnd > pStart and pEnd >= 0 and pStart >= 0: #end not >= start (greater equal), since we look at diagonals and need two non-equal values
        proteins = pProteins[pStart:pEnd].reset_index(drop=False)

        numberOfProteins = proteins.shape[1] - 1

        # Get those indices and corresponding read values of the HiC-matrix that shall be used 
        # for learning and predicting.
        # Since HiC matrices are symmetric, looking at the upper triangular matrix is sufficient
        # It has been shown that taking the full triangle is not good, since most values 
        # are zero or close to zero. So, take the indices of the main diagonal 
        # and the next pWindowSize-1 side diagonals. This structure is a trapezoid
        numberOfDiagonals = min(pEnd-pStart,pWindowSize) #range might be smaller than window size, if centromere close to start / end of chromosome or if valid region small
        trapezIndices = np.mask_indices(proteins.shape[0],maskFunc,k=numberOfDiagonals)
        
        reads = None
        readMatrix = None
        if pFullReads != None:
            readMatrix = pFullReads[pStart:pEnd,pStart:pEnd]
            reads = np.array(readMatrix[trapezIndices])[0] # get only the relevant reads

        dsList = []
        cols = ['first', 'second', 'chrom', 'startProt', 'middleProt', 'endProt', 'distance', 'reads', 'proteinNr']
        for protein in tqdm(range(numberOfProteins)):
            protDf = pd.DataFrame(columns=cols,dtype=np.uint32)
            protDf['first'] = np.uint32(trapezIndices[0])
            protDf['second'] = np.uint32(trapezIndices[1])
            protDf['distance'] = protDf['second'] - protDf['first']
            protDf['chrom'] = np.uint8(pChrom)
            if pFullReads != None:
                protDf['reads'] = np.float32(reads)
                protDf['reads'].fillna(0, inplace=True)
            protDf['proteinNr'] = np.uint8(protein)
            
            #get the protein values for the row ("first") and column ("second") position
            #in the HiC matrix
            protIndex = str(protein)
            startProts = list(proteins[protIndex][protDf['first']])
            protDf['startProt'] = startProts
            endProts = list(proteins[protIndex][protDf['second']])
            protDf['endProt'] = endProts

            #compute window proteins for all positions ending at "second"
            #and all window sizes between 1 and pWindowSize
            #windowDf[x,y] = middle protein values for "second" = x and "distance" = y
            protDf['middleProt'] = 0.
            winSize = min(pEnd-pStart, pWindowSize)
            windowDf = buildWindowDataset(proteins, protein, winSize, pWindowOperation)
        
            #get the window proteins into an array and slice it to get all values at once 
            #there might be a more efficient way using pandas
            distWindowArr = windowDf.to_numpy()
            slice1 = list(protDf['second'])
            slice2 = list(protDf['distance'])
            slice3 = (slice1, slice2)
            windowProteins = np.array(distWindowArr[slice3])
            protDf['middleProt'] = np.float32(windowProteins)

            dsList.append(protDf)
        
        df = pd.concat(dsList, ignore_index=True, sort=False)
        df['first'] += pStart
        df['second'] += pStart

        #drop all rows where start and end protein are both zero
        mask1 = df['startProt'] > 0 
        mask2 = df['endProt'] > 0
        #print("rows before: ", df.shape[0])
        df = df[mask1 & mask2]
        #print("rows after: ", df.shape[0])
    return df

def get_ranges(starts,ends):

    """
    calculating the correct indices for adding up the proteins
    even though the window sizes change
    Attributes: 
        starts -- start positions of bins
        ends --  end positions of bins

    """

    counts = ends - starts
    counts[counts< 0] = 0
    counts_csum = counts.cumsum()
    id_arr = np.ones(counts_csum[-1],dtype=int)
    id_arr[0] = starts[0]
    id_arr[counts_csum[:-1]] = starts[1:] - ends[:-1] + 1
    return id_arr.cumsum()

def getMiddle(proteins,starts,ends, windowOperation):
    """
    Get all indices and the IDs corresponding to same groups
    calculate the window values for each protein
    Attributes:
        proteins -- protein array
        starts -- start positions of bins
        ends --  end positions of bins
        windowOperation --  window bin operation
    """
    idx = get_ranges(starts,ends)
    counts = ends - starts
    counts[counts< 0] = 0
    id_arr = np.repeat(np.arange(starts.size),counts)
    ### use right and left shift to mask all the values we do not want to sum
    right_shift = id_arr[:-2]
    left_shift = id_arr[2:]
    mask = left_shift - right_shift + 1
    mask[mask != 1] = 0
    mask = np.insert(mask,0,[0])
    mask = np.append(mask,[0])
    grp_counts = np.bincount(id_arr) -2
    grp_counts[grp_counts < 1] = 1
    for i in range(1, len(proteins)):
        ### for  each protein sum up all of the windows according to window bin
        ### operation
        slice_arr = proteins[i][idx]
        slice_arr = slice_arr *  mask
        bin_count = np.bincount(id_arr,slice_arr)
        if windowOperation == "avg":
            yield bin_count/grp_counts
        elif windowOperation == "sum":
            yield bin_count
        # elif windowOperation == "max":
            # yield bin_count

def maskFunc(pArray, pWindowSize=0):
    maskArray = np.zeros(pArray.shape)
    upperTriaInd = np.triu_indices(maskArray.shape[0])
    notRequiredTriaInd = np.triu_indices(maskArray.shape[0], k=pWindowSize)
    maskArray[upperTriaInd] = 1
    maskArray[notRequiredTriaInd] = 0
    return maskArray

def buildWindowDataset(pProteinsDf, pProteinNr, pWindowSize, pWindowOperation):
    df = pd.DataFrame()
    proteinIndex = str(pProteinNr)
    for winSize in range(pWindowSize):
        if pWindowOperation == "max":
            windowColumn = pProteinsDf[proteinIndex].rolling(window=winSize+1).max()
        elif pWindowOperation == "sum":
            windowColumn = pProteinsDf[proteinIndex].rolling(window=winSize+1).sum()
        else:
            windowColumn = pProteinsDf[proteinIndex].rolling(window=winSize+1).mean()
        df[str(winSize)] = windowColumn.round(3).astype('float32')
    return df

def findValidProteinRegions(pProteins, pLenThreshold):
    ###filter out regions of length larger than pLenThreshold
    ###where none of the proteins has a peak
    ###return start and end incdices of valid regions
    maxIndex = pProteins.shape[0]-1
    validStartIndices = []
    validEndIndices = []
    #remove unecessary columns and sum over the remaining protein peak columns
    #window column is 0, if none of the proteins has any peak within the (forward facing) window of length pLenThreshold
    clearedProts = pProteins
    clearedProts['sum'] = clearedProts.sum(axis=1)
    clearedProts['window'] = clearedProts['sum'].rolling(window=pLenThreshold).max()
    
    #there are hopefully more valid then invalid regions, so it makes
    #sense to filter for invalid regions and compute valid regions from there
    invalidMask = clearedProts['window'] <= 1.0
    endList = list( clearedProts[invalidMask].index ) 
    if len(endList) == 0:
        #no invalid regions, i. e. whole chromosome is valid
        validStartIndices = [0]
        validEndIndices = [maxIndex]
    elif len(endList) > 0 and len(endList) < pProteins.shape[0]:
        startList = endList[1:] + [endList[-1]+1] #endList[1:] = [] for lists of length 1
        invalidStartIndices = [endList[0]-pLenThreshold+1]
        invalidEndIndices = []
        for end, start in zip(endList, startList):
            if start - end > 1:
                invalidEndIndices.append(end)
                invalidStartIndices.append(start-pLenThreshold+1)
        invalidEndIndices.append(endList[-1])
        validStartIndices = [x+1 for x in invalidEndIndices if x < maxIndex]
        validEndIndices = [x-1 for x in invalidStartIndices if x > 0 ]
        if invalidStartIndices[0] != 0:
            validStartIndices.insert(0, 0)
        if invalidEndIndices[-1] != maxIndex:
            validEndIndices.append(maxIndex) 
    #drop the sum and max columns (since we've not copied the protein df)
    clearedProts.drop(columns=['sum', 'window'], inplace=True)
    return (validStartIndices, validEndIndices)
    
def smoothenProteins(pProteins, pSmooth):
    smoothenedProtDf = pd.DataFrame(columns=pProteins.columns)
    for column in pProteins.columns:
        #compute window size. 
        winSize = int(8*pSmooth) #try a window width of 8 sigma - 4 sigma on both sides
        if winSize % 2 == 0:
            winSize += 1 #window size should not be even, shifting the input otherwise
        winSize = max(3, winSize) #window size should be at least 3, otherwise no smoothing
        #use gaussian for smoothing
        smoothenedProtDf[column] = pProteins[column].rolling(window=winSize, win_type='gaussian', center=True).mean(std=pSmooth)
    smoothenedProtDf.fillna(method='bfill', inplace=True) #fill first (window/2) bins
    smoothenedProtDf.fillna(method='ffill', inplace=True) #fill last (window/2) bins
    return smoothenedProtDf


if __name__ == '__main__':
    createTrainingSet()