#!/usr/bin/env python3
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import click
import hicprediction.configurations as conf
from hicprediction.tagCreator import createSetTag, createProteinTag, initParamDict
from pkg_resources import resource_filename
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import h5py
import joblib
from hicmatrix import HiCMatrix as hm
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

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
def createTrainingSet(chromosomes, datasetoutputdirectory, basefile,\
                      normalizeproteins, normsignalvalue, normsignalthreshold, 
                     divideproteinsbymean,
                    normalizereadcounts, normcountvalue, normcountthreshold,
                   internalindir, windowoperation, mergeoperation, 
                   windowsize, smooth, method, removeempty, printproteins):
    """
    Wrapper function
    calls function and can be called by click
    """
    
    #sanity check of normalization params
    if normalizeproteins and normsignalvalue <= normsignalthreshold:
        msg = "Aborting. \n" 
        msg += "normSignalThreshold must be (much) smaller than normSignalValue"
        raise SystemExit(msg)
    if normalizereadcounts and normcountvalue <= normcountthreshold:
        msg = "Aborting. \n"
        msg += "normCountThreshold must be (much) smaller than normCountValue"
        raise SystemExit(msg)

    createTrainSet(chromosomes= chromosomes,\
                   datasetoutputdirectory= datasetoutputdirectory,\
                   basefile= basefile,\
                   pNormalizeProteins= normalizeproteins,\
                   pNormSignalValue= normsignalvalue,\
                   pNormSignalThreshold= normsignalthreshold,\
                   pDivideProteinsByMean= divideproteinsbymean,\
                   pNormalizeReadCounts= normalizereadcounts,\
                   pNormCountValue= normcountvalue,\
                   pNormCountThreshold= normcountthreshold,\
                   internalInDir= internalindir, \
                   pWindowOperation= windowoperation,\
                   pMergeOperation= mergeoperation, \
                   pWindowsize= windowsize, \
                   pSmooth= smooth,\
                   pMethod= method, \
                   pRemoveEmpty= removeempty, \
                   pPrintProteins= printproteins)

def createTrainSet(chromosomes, datasetoutputdirectory,basefile,
                   pNormalizeProteins, pNormSignalValue, pNormSignalThreshold,
                   pDivideProteinsByMean,
                   pNormalizeReadCounts, pNormCountValue, pNormCountThreshold,
                   internalInDir, pWindowOperation, pMergeOperation, 
                   pWindowsize, pSmooth, pMethod, pRemoveEmpty,
                   pPrintProteins):
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
        params = initParamDict()
        params['chrom'] = chromTag
        params['windowOperation'] = pWindowOperation
        params['mergeOperation'] = pMergeOperation
        params['normalize'] = pNormalizeProteins
        params['normSignalValue'] = pNormSignalValue
        params['normSignalThreshold'] = pNormSignalThreshold
        params['normReadCount'] = pNormalizeReadCounts
        params['normReadCountValue'] = pNormCountValue
        params['normReadCountThreshold'] = pNormCountThreshold
        params['windowSize'] = pWindowsize
        params['removeEmpty'] = pRemoveEmpty
        params['divideProteinsByMean'] = pDivideProteinsByMean
        proteinTag = createProteinTag(params)
        proteinChromTag = proteinTag + "_" + chromTag
        ### retrieve proteins and parameters from base file
        with pd.HDFStore(basefile) as store:
            proteins = store[proteinChromTag]
            params2 = store.get_storer(proteinChromTag).attrs.metadata
        for key, value in params2.items():
            if params[key] == None and key in params2 and params2[key]:
                params[key] = value
        ###smoothen the proteins by gaussian filtering, if desired
        if pSmooth > 0.0:
            proteins = smoothenProteins(proteins, pSmooth)
            params['smoothProt'] = pSmooth
        ###normalize the proteins, if desired
        for protein in range(proteins.shape[1]):
            if pDivideProteinsByMean:
                meanVal = proteins[str(protein)].mean()
                if meanVal <= 1e-6:
                    meanVal = 1.
                    print("Warning: mean signal value < 1e6, check protein input nr. {:d}".format(protein + 1))
                proteins[str(protein)] /= meanVal
            if pNormalizeProteins and pNormSignalValue > 0.0: 
                scaler = MinMaxScaler(feature_range=(0, pNormSignalValue), copy=False)
                proteins[[str(protein)]] = scaler.fit_transform(proteins[[str(protein)]])
                thresMask = proteins[str(protein)] < pNormSignalThreshold
                proteins.loc[thresMask, str(protein)] = 0.0
        if pDivideProteinsByMean:
            print("Divided each protein by mean")
        if pNormalizeProteins:
            msg = " normalized protein signal values to range 0...{:.2f}\n"
            msg += " Set values < {:.3f} to zero"
            msg = msg.format(pNormSignalValue, pNormSignalThreshold)
            print("Protein normalization:")
            print(msg)
        ###print some info about the proteins:
        print("non-zero entries:")
        for protein in range(proteins.shape[1]):
            nzmask = proteins[str(protein)] > 0.
            nonzeroEntries = proteins[nzmask].shape[0]
            protMin = proteins.loc[nzmask, str(protein)].min()
            protMax = proteins.loc[nzmask, str(protein)].max()
            msg = "protein {:d}: {:d} of {:d} (min: {:.3f}, max: {:.3f})"
            msg = msg.format(protein, nonzeroEntries, proteins.shape[0], \
                    protMin, protMax)
            print(msg)
            if pPrintProteins:
                xvals = list(proteins.index.astype('int32')) 
                yvals = list(proteins[str(protein)])
                xvals = [x*int(params['resolution'])/1e6 for x in xvals]
                #ensure resolution is sufficient to show all bars
                figwidthInches = 15
                columnWidthInches = (figwidthInches - 1) * (int(params['resolution'])/1e6) / max(xvals) #subtract 1in as a safety margin
                figDpi = int(math.ceil(1/(columnWidthInches * 100.0))) * 100 #round to nearest 100 dpi
                fig1, ax1 = plt.subplots(figsize=(figwidthInches,3), dpi=figDpi, constrained_layout=True)
                ax1.bar(xvals, yvals, width=int(params['resolution'])/1e6)
                ax1.set_xlim([min(xvals), max(xvals)])
                ax1.set_xlabel('genomic position / Mbp')
                ax1.set_ylabel('signal value')
                ax1.set_title(params['proteinFileNames'][protein])
                figname = str(params['cellType']) + '_protein_' + str(protein) + '.png' 
                fig1.savefig(os.path.join(datasetoutputdirectory, figname))
                
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
            createDataset = createDatasetOneHot
        else:
            createDataset = createDatasetMultiColumn
        params['method'] = pMethod

        ### create datasets with read counts and proteins
        fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1, constrained_layout=True)
        df = createDataset(pProteins=proteins, pFullReads=reads, 
                            pWindowOperation=pWindowOperation, pWindowSize=pWindowsize,
                            pChrom=chromosome, pStart=0, pEnd=proteins.shape[0]-1, pAxis=ax1, pRemoveEmpty=pRemoveEmpty)
        if df.empty:
            msg = "Could not create dataset. Aborting"
            raise SystemExit(msg)
        
        #print some figures
        validMask = df['valid'] == True
        print( "{0:d} valid samples in dataset".format(df[validMask].shape[0]) )
        readNonzeroMask = df['reads'] > 0
        print( "{:d} valid samples with read count > 0".format(df[validMask & readNonzeroMask].shape[0]) )

        if reads != None:
            binWidth = 10
            nrBins = int(np.round(df[validMask]['reads'].max()/binWidth))
            df[validMask]['reads'].plot.hist(bins=nrBins, ax=ax2)
            ax2.set_yscale('log')
            ax2.set_yticks([1,10,1e2,1e3,1e4,1e5])
            ax2.set_title("Read count distribution after invalidating places without assoc. proteins")
            ax2.set_xlim(ax1.get_xlim()) 
            ax2.set_ylim(ax1.get_ylim())

        ### normalize ALL read counts
        if pNormalizeReadCounts and pNormCountValue > 0.0:
            scaler = MinMaxScaler(feature_range=(0, pNormCountValue), copy=False)
            df[['reads']] = scaler.fit_transform(df[['reads']])
            thresMask = df['reads'] < pNormCountThreshold
            df.loc[thresMask, 'reads'] = 0.0
            msg = "normalized all read counts to range 0...{:.2f}\n"
            msg += "Set values < {:.3f} to zero"
            msg = msg.format(pNormCountValue, pNormCountThreshold)
            print(msg)

        ### add average contact read stratified by distance to dataset
        if matrixfile:
            for i in tqdm(range(int(pWindowsize)),desc='Adding average read values'):
                df.loc[df['distance'] == i,'avgRead'] =  df[df['distance'] == i]['reads'].mean()
            
        #one-hot encoding for the proteins / protein numbers
        if pMethod == 'oneHot':
            df['proteinNr'] = df['proteinNr'].astype('category')
            df = pd.get_dummies(df, prefix='prot')

        #finally, store the dataset  
        setTag = createSetTag(params) + ".z"
        datasetFileName = os.path.join(datasetoutputdirectory, setTag)  
        joblib.dump((df, params), datasetFileName,compress=True ) 

        #plot read count distribution
        if reads != None:
            df[validMask]['reads'].plot.hist(bins=100, ax=ax3)
            ax3.set_yscale('log')
            ax3.set_title("Final read count distribution in dataset")
            ax3.set_yticks([1,10,1e2,1e3,1e4,1e5])
            ax3.set_ylim(ax1.get_ylim())
            rcDistributionFilename = createSetTag(params) + "_rcDistribution.png"
            fig1.suptitle("Read count distributions {:s}, {:s}".format(params['cellType'], params['chrom']))
            fig1.savefig(os.path.join(datasetoutputdirectory, rcDistributionFilename))

def createDatasetMultiColumn(pProteins, pFullReads, pWindowOperation, pWindowSize,
                   pChrom, pStart, pEnd, pAxis, pRemoveEmpty):
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
    
            #read count distribution before eliminating zeros
            binwidth = 10
            nrBins = int(np.round(df['reads'].max()/binwidth))
            df['reads'].plot.hist(bins=nrBins, ax=pAxis)
            pAxis.set_yscale('log')
            pAxis.set_yticks([1,10,1e2,1e3,1e4,1e5])
            pAxis.set_title("Read count distribution from input (cooler)")

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

        #drop rows where start / end proteins are both zero
        df['valid'] = True
        if pRemoveEmpty:
            df['valid'] = False
            mask = False 
            m1 = False
            m2 = False   
            for i in tqdm(range(numberOfProteins), desc="invalidating rows without start/end proteins"):    
            #    m1 = df[str(i)] > 0
            #    m2 = df[str(numberOfProteins * 2 + i)] > 0
            #    mask = mask | (m1 & m2)
                m1 |= df[str(i)] > 0
                m2 |= df[str(numberOfProteins * 2 + i)] > 0
            mask = m1 & m2
            df.loc[mask, 'valid'] = True
            print()
            print( "invalidated {0:d} rows".format(df[~mask].shape[0]) )
        
        #consider offset
        df['first'] += pStart
        df['second'] += pStart

    return df

def createDatasetOneHot(pProteins, pFullReads, pWindowOperation, pWindowSize,
                   pChrom, pStart, pEnd, pAxis, pRemoveEmpty):
    
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
                #read count distribution before eliminating zeros
                binwidth = 10
                nrBins = int(np.round(df['reads'].max()/binwidth))
                df['reads'].plot.hist(bins=nrBins, ax=pAxis)
                pAxis.set_yscale('log')
                pAxis.set_title("Read count distribution from matrix")
                pAxis.set_yticks([1,10,1e2,1e3,1e4,1e5])
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

        #invalidate all rows where start and end protein are both zero
        df['valid'] = True
        if pRemoveEmpty == True:
            df['valid'] = False
            mask1 = df['startProt'] > 0 
            mask2 = df['endProt'] > 0
            mask = mask1 & mask2
            df.loc[mask, 'valid'] = True
            print()
            print( "invalidated {0:d} rows".format(df[~mask].shape[0]) )
    return df

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