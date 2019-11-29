#!/usr/bin/env python3

from configurations import *
from hicprediction.tagCreator import createSetTag, createProteinTag

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

@set_options
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
        # centromeresfile =  os.path.dirname(__file__) +\
        # "/InternalStorage/centromeres.txt"
        centromeresfile = resource_filename('hicprediction',\
                'InternalStorage') +"/centromeres.txt"

    checkExtension(basefile, 'ph5')
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
            msg = 'Chromosome {} is not in the range of 1 to 22. Please' +\
                    'provide correct chromosomes'
            print(msg.format(str(chromosome)))
            sys.exit()
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
        setTag =createSetTag(params)
        datasetFileName = datasetoutputdirectory +"/" + setTag + ".z"
        fileExists = os.path.isfile(datasetFileName)
        dir_list = next(os.walk(datasetoutputdirectory))[1]
        for path in dir_list:
            tmpPath = datasetoutputdirectory +"/"+path+"/" + setTag + ".z"
            fileExists = fileExists or os.path.isfile(tmpPath)
        if not fileExists:
            rows = np.shape(proteins)[0]
            with h5py.File(basefile, 'r') as baseFile:
                if chromTag not in baseFile:
                    msg = 'The chromosome {} is not loaded yet. Please'\
                            +'update your chromosome file {} using the script'\
                            +'"getChroms"'
                    sys.exit()
                ### load HiC matrix
                matrixfile = baseFile[chromTag].value
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
                start, end = getCentromerePositions(centromeresfile, chromTag,cuts)
                for i in tqdm(range(2), desc = "Loading chromatids separately"):
                    if i == 0:
                        df = createDataset(proteins, reads, windowoperation, windowsize,
                                chromosome, start=0, end=start)
                    elif i == 1:
                        df = df.append(createDataset(proteins, reads,\
                                windowoperation, windowsize,\
                                chromosome, start=end + 1, end=len(cuts)))
            else:
                df = createDataset(proteins, reads, windowoperation, windowsize,
                               chromosome, start=0, end =len(cuts))
            ### add average contact read stratified by distance to dataset
            for i in range(int(windowsize)):
                df.loc[df['distance'] == i,'avgRead'] =  df[df['distance'] == i]['reads'].mean()
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
    if windowOperation == 'avg':
        convertF = np.mean
    elif windowOperation == 'sum':
        convertF = np.sum
    elif windowOperation == 'max':
        convertF = np.max
    
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


if __name__ == '__main__':
    createTrainingSet()
