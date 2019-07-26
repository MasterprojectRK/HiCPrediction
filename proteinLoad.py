from configurations import *
from tagCreator import tagCreator

def chrom_filter(feature, c):
        return feature.chrom == c

def loadAllProteins(args, armDict):
    column = 4
    proteins = dict()
    for f in os.listdir(PROTEIN_D):
        if f.startswith(args.cellLine):
            path  = PROTEIN_D+f
            a = pybedtools.BedTool(path)
            b = a.to_dataframe()
            c = b.iloc[:,column]
            minV = min(c)
            maxV = max(c) - minV
            if maxV == 0:
                maxV = 1
            if args.normalizeProteins:
                for row in a:
                    row[column] = (float(x[column]) - minV) / maxV
            proteins[f] = a
    proteinDict = dict()
    for name, arm in tqdm(armDict.items(), desc= 'Converting proteins for each arm'):
        proteinDict[name] = loadProtein(args, proteins, name, arm)
    return proteinDict


def loadProtein(args, proteins, armName, arm):
    chromName = armName.split("_")[0]
    cuts = arm.cut_intervals
    i = 0
    allProteins = []
    if args.mergeOperation == 'avg':
        merge = np.mean
    elif args.mergeOperation == 'max':
        merge = np.max
    elif args.mergeOperation == 'sum':
        merge = np.sum
    for cut in cuts:
        allProteins.append(np.zeros(15))
        allProteins[i][0] = cut[1]
        i += 1
    cutsReduced = list(map(lambda x: x[1], cuts))
    i = 0
    for name, a in tqdm(proteins.items(), desc = 'Proteins converting'):
        a = a.filter(chrom_filter, c='chr'+ chromName)
        a = a.sort()
        values = dict()
        for feature in a:    
            peak = feature.start + int(feature[9])
            pos = bisect.bisect_right(cutsReduced, peak)
            if pos in values:
                values[pos].append(float(feature[6]))
            else:
                values[pos] = [float(feature[6])]
        j = 0
        for key, val in values.items():
            score = merge(val)
            allProteins[key - 1][i+1] = score 
        i += 1
    data = pd.DataFrame(allProteins,columns=['start','ctcf', 'rad21', 'smc3',\
                        'H2az', 'H3k4me1', 'H3k4me2', 'H3k4me3','H3k9ac',\
                        'H3k9me3','H3k27ac', 'H3k27me3','H3k36me3',\
                        'H3k79me2', 'H4k20me1'], index=range(len(cuts)))
    return data
