#!/usr/bin/env python3

from configurations import *
from tagCreator import createSetTag

@standard_options
@click.argument('proteinInputFiles',nargs=-1)
@click.command()
def createTrainSet(centromeresfile, proteininputfiles,\
                    experimentoutputdirectory,windowoperation,windowsize,\
                    ignorecentroids, equalize, chromosomes):
    if chromosomes:
        chromosomeList = chromosomes.split(',')
    else:
        chromosomeList = range(1, 23)
    for proteinFile in tqdm(proteininputfiles):
        chromosomeFileName = experimentoutputdirectory + '/chromosomes.ch5'
        if not os.path.isdir(experimentoutputdirectory + "/Sets"):
            os.mkdir(experimentoutputdirectory + "/Sets")
        if not os.path.isfile(chromosomeFileName):
            addGenome(matrixfile, chromosomeFileName)
        proteinTag = proteinFile.split("/")[-1].split(".")[0]
        with h5py.File(chromosomeFileName, 'a') as chromFile:
            for chromosome in tqdm(chromosomeList):
                chromosome = int(chromosome)
                if not chromosome in range(1,23):
                    msg = 'Chromosome {} is not in the range of 1 to 22. Please' +\
                            'provide correct chromosomes'
                    print(msg.format(str(chromosome)))
                    sys.exit()
                chromTag = "chr" +str(chromosome)
                setTag =createSetTag(proteinTag,chromTag, window=\
                        windowoperation,eq=equalize,ignore=ignorecentroids,
                                    windowSize=windowsize)
                try:
                    proteins = pd.read_hdf(proteinFile,key=chromTag, mode='r')
                except KeyError:
                    msg = 'Key {} does not exist in Protein file {}. Please' +\
                            'execute the script "proteinLoad" with the correct ' +\
                            'parameters'
                    print(msg.format(chromTag, proteinFile))
                    sys.exit()
                rows = np.shape(proteins)[0]
                if chromTag not in chromFile:
                    msg = 'The chromosome {} is not loaded yet. Please'\
                            +'update your chromosome file {} using the script'\
                            +'"getChroms"'
                    sys.exit()
                readPath = chromTag+"/pixels/"
                bin1_id = chromFile[readPath+"bin1_id"].value
                bin2_id = chromFile[readPath+"bin2_id"].value
                counts = chromFile[readPath+"count"].value
                cuts = chromFile[chromTag + "/bins/start"].value
                reads = coo_matrix((counts, (bin1_id, bin2_id)), shape=(rows, rows))

                if ignorecentroids and chromosome not in [13,14,15,22]:
                    start, end = getTransArmPositions(centromeresfile, chromTag,cuts)
                    df = createDataset(proteins, reads, windowoperation, windowsize,
                                   equalize, chromosome, start=0, end=start)
                    df = df.append(createDataset(proteins, reads, windowoperation, windowsize,
                                   equalize, chromosome, start=end + 1, end=len(cuts)))
                else:
                    df = createDataset(proteins, reads, windowoperation, windowsize,
                                   equalize, chromosome, start=0, end =len(cuts))
                fileName = experimentoutputdirectory +'/Sets/' +setTag+'.brotli'
                df.to_parquet(fileName, engine='auto')

def getTransArmPositions(centromeresfilepath, chromTag, cuts):
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
                  equalize, chrom, start, end):
    proteins = proteins[start:end]
    colNr = np.shape(proteins)[1]
    proteinNr = colNr - 1
    rows = np.shape(proteins)[0]
    fullReads = fullReads.todense()
    reads = fullReads[start:end, start:end]
    strList = [str(x) for x in range(3*(proteinNr))]
    cols = ['first', 'second','chrom'] + strList+['distance','reads']

    if windowOperation == 'avg':
        convertF = np.mean
    elif windowOperation == 'sum':
        convertF = np.sum
    elif windowOperation == 'max':
        convertF = np.max

    reach  = int(windowSize)
    height = int((rows - reach) * reach +  reach * (reach +1) /2)
    lst = range(rows - reach)
    idx1 = list(itertools.chain.from_iterable(itertools.repeat(x, reach) for x in lst))
    for i in range(reach):
        idx1.extend((reach - i) * [rows - reach + i])
    idx2 =[]
    for i in range(rows - reach):
        idx2.extend(list(range(i, i + reach)))
    for i in range(rows - reach, rows):
        idx2.extend(list(range(i, rows)))
    df = pd.DataFrame(0, index=range(height), columns=cols)
    df['chrom'] = str(chrom)
    df['first'] = np.array(idx1)
    df['second'] = np.array(idx2)
    df['distance'] = (df['second'] - df['first']) 
    df['reads'] = np.array(reads[df['first'],df['second']])[0]

    proteinMatrix = np.matrix(proteins)
    starts = df['first'].values
    ends = df['second'].values + 1
    middleGenerator = getMiddle(proteins.values.transpose(), starts, ends,
                                windowOperation)
    for i in tqdm(range(proteinNr), desc="Converting Proteins to dataset"):
        df[str(i)] = np.array(proteinMatrix[df['first'], i+1]).flatten()
        df[str(proteinNr + i)] = next(middleGenerator) 
        df[str(proteinNr * 2 + i)] = np.array(proteinMatrix[df['second'], i+1]).flatten()
    df['first'] += start
    df['second'] += start
    return df

def createAllCombinations(args):
    for w in ["avg"]:
        args.windowOperation = w
        for m in ["avg"]:
            args.mergeOperation = m
            for n in [False]:
                args.normalizeProteins = n
                for e in [False]:
                    args.equalizeProteins = e
                    createAllWindows(args)


def get_ranges_arr(starts,ends):
    counts = ends - starts
    counts[counts< 0] = 0
    counts_csum = counts.cumsum()
    id_arr = np.ones(counts_csum[-1],dtype=int)
    id_arr[0] = starts[0]
    id_arr[counts_csum[:-1]] = starts[1:] - ends[:-1] + 1
    return id_arr.cumsum()

def getMiddle(proteins,starts,ends, windowOperation):
    # Get all indices and the IDs corresponding to same groups
    idx = get_ranges_arr(starts,ends)
    counts = ends - starts
    counts[counts< 0] = 0
    id_arr = np.repeat(np.arange(starts.size),counts)
    right_shift = id_arr[:-2]
    left_shift = id_arr[2:]
    mask = left_shift - right_shift + 1
    mask[mask != 1] = 0
    mask = np.insert(mask,0,[0])
    mask = np.append(mask,[0])
    grp_counts = np.bincount(id_arr) -2
    grp_counts[grp_counts < 1] = 1
    for i in range(1,15):
        slice_arr = proteins[i][idx]
        slice_arr = slice_arr *  mask
        bin_count = np.bincount(id_arr,slice_arr)
        if windowOperation == "avg":
            yield bin_count/grp_counts
        elif windowOperation == "sum":
            yield bin_count


if __name__ == '__main__':
    createTrainSet()
