from configurations import *
from tagCreator import tagCreator


def createDataset(args, name, arm, chrom, proteins):
    colNr = np.shape(proteins)[1]
    rows = np.shape(proteins)[0]
    reads = arm.matrix.todense()
    maxValue = chrom.matrix.max()

    cols = ['first', 'second','chrom'] + list(range(3*
         (colNr-1)))+['distance','reads']

    if args.windowOperation == 'avg':
        convertF = np.mean
    elif args.windowOperation == 'sum':
        convertF = np.sum
    elif args.windowOperation == 'max':
        convertF = np.max

    reach  = int(args.reach)
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
    df['chrom'] = name
    df['first'] = np.array(idx1)
    df['second'] = np.array(idx2)
    df['distance'] = (df['second'] - df['first']) * args.binSize
    df['reads'] = np.array(reads[df['first'],df['second']])[0]

    proteinMatrix = np.matrix(proteins)
    proteinFrame =  pd.DataFrame(proteins)
    starts = df['first'].values
    ends = df['second'].values + 1 
    for i in range(14):
        df[i] = np.array(proteinMatrix[df['first'], i+1]).flatten()
        df[14 + i] = getMiddle(proteinFrame, starts, ends, args)
        df[28 + i] = np.array(proteinMatrix[df['second'], i+1]).flatten()
    return df

def multiplyBin(a, b):
    return a * b * args.binSize
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


def createAllWindows(args):
    i = 1
    for f in os.listdir(ARM_D):
        args.chrom = f.split(".")[0]
        if  os.path.isfile(tagCreator(args, "protein")):
            # if int(args.chrom.split("_")[0]) < 22:
                # continue
            print(args.chrom)
            if True: 
            # if  not os.path.isfile(tagCreator(args, "set")):
                createForestDataset(args)

def get_ranges_arr(starts,ends):
    counts = ends - starts
    counts[counts< 0] = 0
    counts_csum = counts.cumsum()
    id_arr = np.ones(counts_csum[-1],dtype=int)
    id_arr[0] = starts[0]
    id_arr[counts_csum[:-1]] = starts[1:] - ends[:-1] + 1
    return id_arr.cumsum()

def getMiddle(proteins,starts,ends, args):
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
        if args.windowOperation == "avg":
            yield bin_count/grp_counts
        elif args.windowOperation == "sum":
            yield bin_count
