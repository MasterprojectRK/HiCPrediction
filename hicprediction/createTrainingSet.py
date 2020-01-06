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
                   internalindir, windowoperation, mergeoperation, windowsize, peakcolumn):
    """
    Wrapper function
    calls function and can be called by click
    """
    createTrainSet(chromosomes, datasetoutputdirectory,basefile,\
                   centromeresfile,ignorecentromeres,normalize,
                   internalindir, windowoperation, mergeoperation, windowsize, peakcolumn)

def createTrainSet(chromosomes, datasetoutputdirectory,basefile,\
                   centromeresfile,ignorecentromeres,normalize,
                   internalInDir, windowoperation, mergeoperation, windowsize, peakcolumn):
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
            peakcolumn -- Column in bed file tha contains peak values
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
        ### check if chromosme is along the 22 first chromosomes
        if not chromosome in range(1,23):
            msg = 'Chromosome {0:d} is not in the range of 1 to 22.'
            msg += 'Please provide correct chromosomes'
            sys.exit( msg.format(chromosome) )
        ### create parameter set
        params = dict()
        params['chrom'] = chromTag
        params['windowOperation'] = windowoperation
        params['mergeOperation'] = mergeoperation
        params['normalize'] = normalize
        params['ignoreCentromeres'] = ignorecentromeres
        params['peakColumn'] = peakcolumn
        params['windowSize'] = windowsize
        proteinTag =createProteinTag(params)
        proteinChromTag = proteinTag + "_" + chromTag
        ### retrieve proteins and parameters from base file
        with pd.HDFStore(basefile) as store:
            proteins = store[proteinChromTag]
            params2 = store.get_storer(proteinChromTag).attrs.metadata
        ### join parameters
        params = {**params, **params2}
        setTag = createSetTag(params) + ".z"
        datasetFileName = os.path.join(datasetoutputdirectory, setTag)
        fileExists = os.path.isfile(datasetFileName)
        dir_list = next(os.walk(datasetoutputdirectory))[1]
        for path in dir_list:
            tmpPath = os.path.join(datasetoutputdirectory, path, setTag)
            fileExists = fileExists or os.path.isfile(tmpPath)
        fileExists = False
        if not fileExists:
            with h5py.File(basefile, 'r') as baseFile:
                if chromTag not in baseFile:
                    msg = 'The chromosome {} is not loaded yet. Please'\
                            +'update your chromosome file {} using the script'\
                            +'"getChroms"'
                    sys.exit()
                ### load HiC matrix
                matrixfile = baseFile[chromTag][()]
                if internalInDir:
                    filename = os.path.basename(matrixfile)
                    matrixfile = os.path.join(internalInDir, filename)
                hiCMatrix = None
                if os.path.isfile(matrixfile):
                    hiCMatrix = hm.hiCMatrix(matrixfile)
                else:
                    msg = ("cooler file {0:s} is missing.\n" \
                          + "Use --iif option to provide the directory where the internal matrices " \
                          +  "were stored when creating the basefile").format(matrixfile)
                    sys.exit(msg)
                
            ### load reads and bins
            reads = hiCMatrix.matrix
            cuts = hiCMatrix.cut_intervals
            cuts = np.array([cut[1] for cut in cuts])
            ### if user decided to cut out centromeres and if the chromosome
            ### has one, create datasets for both chromatids and join them
            if ignorecentromeres :
                # start, end = getCentromerePositions(centromeresfile, chromTag,cuts)
                # for i in tqdm(range(2), desc = "Loading chromatids separately"):
                #     if i == 0:
                #         df = createDataset2(proteins, reads, windowoperation, windowsize,
                #                 chromosome, pStart=0, pEnd=start)
                #     elif i == 1:
                #         df = df.append(createDataset2(proteins, reads,\
                #                 windowoperation, windowsize,\
                #                 chromosome, pStart=end + 1, pEnd=len(cuts)), ignore_index=True, sort=False)
                threshold = int(1000000 / int(params['resolution']))
                starts, ends = findValidProteinRegions(proteins, threshold)
                if not starts or not ends or len(starts) < 1 or len(ends) < 1:
                    msg = "No valid protein peaks found. Aborting."
                    sys.exit(msg)
                else:
                    dfList = []
                    for s, e in zip(starts, ends):
                        dfList.append(createDataset2(proteins, reads, windowoperation, windowsize,
                         chromosome, pStart = s, pEnd = e))
                    df = pd.concat(dfList, ignore_index=True, sort=False)     
            else:
                df = createDataset2(proteins, reads, windowoperation, windowsize,
                               chromosome, pStart=0, pEnd=len(cuts))
            if df.empty:
                msg = "Could not create dataset. Aborting"
                sys.exit(msg)

            ### add average contact read stratified by distance to dataset
            for i in tqdm(range(int(windowsize)),desc='Adding average read values'):
                df.loc[df['distance'] == i,'avgRead'] =  df[df['distance'] == i]['reads'].mean()
            
            #one-hot encoding for the proteins / protein numbers
            df['proteinNr'] = df['proteinNr'].astype('category')
            df = pd.get_dummies(df, prefix='prot')
            #print(df.head(10))
            #print(df.tail(10))
            
            joblib.dump((df, params), datasetFileName,compress=True ) 
        else:
            print("Skipped creating file that already existed: " + datasetFileName)
    print("\n")

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

def createDataset(proteins, fullReads, windowOperation, windowSize,
                   chrom, start, end):
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

    ### cut proteins according to centromere positions
    proteins = proteins[start:end]
    colNr = np.shape(proteins)[1]
    proteinNr = colNr - 1
    rows = np.shape(proteins)[0]
    fullReads = fullReads.todense()
    reads = fullReads[start:end, start:end]
    ### create columns of dataset
    strList = [str(x) for x in range(3*(proteinNr))]
    cols = ['first', 'second','chrom'] + strList+['distance','reads']
    ### get window bin operation
    #if windowOperation == 'avg':
    #    convertF = np.mean
    #elif windowOperation == 'sum':
    #    convertF = np.sum
    #elif windowOperation == 'max':
    #    convertF = np.max
    
    reach  = int(windowSize)
    height = int((rows - reach) * reach +  reach * (reach +1) /2)
    lst = range(rows - reach)
    ### compute indices for all the possible contact pairs considering maximum
    ### genomic distance etc.
    idx1 = list(itertools.chain.from_iterable(itertools.repeat(x, reach) for x in lst))
    for i in range(reach):
        idx1.extend((reach - i) * [rows - reach + i])
    idx2 =[]
    for i in range(rows - reach):
        idx2.extend(list(range(i, i + reach)))
    for i in range(rows - reach, rows):
        idx2.extend(list(range(i, rows)))
    ### create data frame and fill with values and indices
    df = pd.DataFrame(0, index=range(height), columns=cols)
    df['chrom'] = str(chrom)
    df['first'] = np.array(idx1)
    df['second'] = np.array(idx2)
    df['distance'] = (df['second'] - df['first']) 
    df['reads'] = np.array(reads[df['first'],df['second']])[0]
    ### load proteins
    proteinMatrix = np.matrix(proteins)
    starts = df['first'].values
    ends = df['second'].values + 1
    ### create generator that will compute the window proteins
    middleGenerator = getMiddle(proteins.values.transpose(), starts, ends,
                                windowOperation)
    ### iterate over all the proteins and fill the data frame
    for i in tqdm(range(proteinNr), desc="Converting Proteins to dataset"):
        # print(i)
        # start = time.time()*1000
        # print(start - time.time()*1000)
        ### Could maybe iterate genomic distance instead of middleGenerator to speed
        ### things up
        df[str(i)] = np.array(proteinMatrix[df['first'], i+1]).flatten()
        df[str(proteinNr + i)] = next(middleGenerator) 
        df[str(proteinNr * 2 + i)] = np.array(proteinMatrix[df['second'], i+1]).flatten()
    df['first'] += start
    df['second'] += start
    return df

def createDataset2(pProteins, pFullReads, pWindowOperation, pWindowSize,
                   pChrom, pStart, pEnd):
    
    df = pd.DataFrame()
    
    if pEnd > pStart and pEnd >= 0 and pStart >= 0: #end not >= start (greater equal), since we look at diagonals and need two non-equal values
        proteins = pProteins[pStart:pEnd].drop('start', axis=1)
        proteins.reset_index(inplace=True, drop=True) #otherwise access indices out of range when ignoring centromeres

        numberOfProteins = proteins.shape[1]

        # Get those indices and corresponding read values of the HiC-matrix that shall be used 
        # for learning and predicting.
        # Since HiC matrices are symmetric, looking at the upper triangular matrix is sufficient
        # It has been shown that taking the full triangle is not good, since most values 
        # are zero or close to zero. So, take the indices of the main diagonal 
        # and the next pWindowSize-1 side diagonals. This structure is a trapezoid
        numberOfDiagonals = min(pEnd-pStart,pWindowSize) #range might be smaller than window size, if centromere close to start / end of chromosome or if valid region small
        readMatrix = pFullReads[pStart:pEnd,pStart:pEnd]
        trapezIndices = np.mask_indices(readMatrix.shape[0],maskFunc,k=numberOfDiagonals)
        reads = np.array(readMatrix[trapezIndices])[0] # get only the relevant reads

        cols = ['first', 'second', 'chrom', 'startProt', 'middleProt', 'endProt', 'distance', 'reads', 'proteinNr']

        dsList = []

        for protein in tqdm(range(numberOfProteins)):
            protDf = pd.DataFrame(columns=cols,dtype=np.uint32)
            protDf['first'] = np.uint32(trapezIndices[0])
            protDf['second'] = np.uint32(trapezIndices[1])
            protDf['distance'] = protDf['second'] - protDf['first']
            protDf['chrom'] = np.uint8(pChrom)
            protDf['reads'] = np.float32(reads)
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
    clearedProts = pProteins.drop('start', axis=1)
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

    return (validStartIndices, validEndIndices)
    

if __name__ == '__main__':
    createTrainingSet()