from configurations import *
from tagCreator import createTag
from scipy.sparse import coo_matrix

@click.group()
def cli():
    pass

@click.option('--equalize/--dont-equalize', default=False)
@click.option('--ignoretransarms/--dont-ignoretransarms', default=True)
@click.option('windowsize', '-ws', default=200)
@click.option('--windowoperation', '-wo', default='avg',\
              type=click.Choice(['avg', 'max', 'sum']))
@click.option('--normalize/--dont-normalize', default=False)
@click.option('--mergeoperation', '-mo', default='avg',\
              type=click.Choice(['avg', 'max']))
@click.option('--cellline', '-cl', default='Gm12878')
@click.option('resolution', '-r', default=5000)
@click.option('proteinfilepath', '-pfp', default='Data/proteins.h5')
@click.option('chromfilepath', '-cfp', default='Data/chroms.h5')
@click.option('centromeresfilepath', '-cmfp',
              default='Data/centromeres.txt')
@click.option('datasetdirectory', '-dsd', default='Data/Sets/')
@cli.command()
# @click.argument('proteindir')
def createAllSets(datasetdirectory, centromeresfilepath,chromfilepath, proteinfilepath,\
                  resolution, cellline, mergeoperation, normalize, windowoperation,\
                  windowsize, ignoretransarms, equalize):

    with h5py.File(chromfilepath, 'a') as chromFile:
        for chrom in tqdm(range(1,23), desc= 'Creating sets for each chromosome'):
            proteinTag =createTag(resolution, cellline, chrom,\
                                  merge=mergeoperation, norm=normalize)
            setTag =createTag(resolution, cellline, chrom,\
                        merge=mergeoperation, norm=normalize,\
                        window=windowoperation, eq=equalize,ignore=ignoretransarms)
            try:
                proteins = pd.read_hdf(proteinfilepath ,key=proteinTag, mode='r')
            except KeyError:
                msg = 'Key {} does not exist in Protein file {}. Please' +\
                        'execute the script "proteinLoad" with the correct ' +\
                        'parameters'
                print(msg.format(proteinTag, proteinfilepath))
                sys.exit()
            except FileNotFoundError:
                msg = 'Given or default protein file {} does not exist. Please' +\
                        ' execute the script "proteinLoad" with the correct ' +\
                        'parameters or set the correct file name'
                print(msg.format(proteinfilepath))
                sys.exit()


            rows = np.shape(proteins)[0]
            chromTag =createTag(resolution, cellline, chrom)
            if chromTag not in chromFile:
                msg = 'The chromosome {} is not loaded yet. Please'\
                        +'update your chromosome file {} using the script'\
                        +'"getChroms"'
                print(msg.format(chromTag, chromfilepath))
                sys.exit()
            readPath = chromTag+"/pixels/"
            bin1_id = chromFile[readPath+"bin1_id"].value
            bin2_id = chromFile[readPath+"bin2_id"].value
            counts = chromFile[readPath+"count"].value
            cuts = chromFile[chromTag + "/bins/start"].value
            reads = coo_matrix((counts, (bin1_id, bin2_id)), shape=(rows,
                                                                   rows))
            if ignoretransarms and chrom not in [13,14,15,22]:
                start, end = getTransArmPositions(centromeresfilepath,
                                                  chrom,cuts)
                df = createDataset(proteins, reads, resolution, windowoperation, windowsize,
                               equalize, chrom, start=0, end=start)
                df.append(createDataset(proteins, reads, resolution, windowoperation, windowsize,
                               equalize, chrom, start=end + 1, end=len(cuts)))
            else:
                df = createDataset(proteins, reads, resolution, windowoperation, windowsize,
                               equalize, chrom, start=0, end =len(cuts))
            df.to_parquet( datasetdirectory +setTag+'.brotli', engine='auto',\
                          compression='brotli')

def getTransArmPositions(centromeresfilepath, chrom, cuts):
        f=open(centromeresfilepath, "r")
        fl =f.readlines()
        elems = None
        for x in fl:
            elems = x.split("\t")
            if elems[1] == "chr"+str(chrom):
                continue
        start = int(elems[2])
        end = int(elems[3])
        toStart = cuts[cuts < start]
        toEnd = cuts[cuts < end]
        return  len(toStart), len(toEnd)

def createDataset(proteins, fullReads, resolution, windowOperation, windowSize,
                  equalize, chrom, start, end):
    proteins = proteins[start:end]
    colNr = np.shape(proteins)[1]
    rows = np.shape(proteins)[0]
    fullReads = fullReads.todense()
    reads = fullReads[start:end, start:end]
    strList = [str(x) for x in range(3*(colNr-1))]
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
    df['distance'] = (df['second'] - df['first']) * resolution
    df['reads'] = np.array(reads[df['first'],df['second']])[0]

    proteinMatrix = np.matrix(proteins)
    starts = df['first'].values
    ends = df['second'].values + 1
    middleGenerator = getMiddle(proteins.values.transpose(), starts, ends,
                                windowOperation)
    for i in tqdm(range(14), desc="Converting Proteins"):
        df[str(i)] = np.array(proteinMatrix[df['first'], i+1]).flatten()
        df[str(14 + i)] = next(middleGenerator) 
        df[str(28 + i)] = np.array(proteinMatrix[df['second'], i+1]).flatten()
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
    cli()
