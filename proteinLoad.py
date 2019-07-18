from configurations import *
from tagCreator import tagCreator

def chrom_filter(feature, c):
        return feature.chrom == c

def peak_filter(feature, s, e):
        peak = feature.start + int(feature[9])
        return s <= peak and e >= peak

def loadAllProteins(args, armDict):
    proteins = dict()
    cell = "Gm12878/"
    for f in os.listdir(PROTEINORIG_D + cell):
        path  = PROTEINORIG_D+cell+f
        a = pybedtools.BedTool(path)
        b = a.to_dataframe()
        c = b.iloc[:,6]
        minV = min(c)
        maxV = max(c) - minV
        if maxV == 0:
            maxV = 1
        if args.normalizeProteins:
            for row in a:
                row[6] = (float(x[6]) - minV) / maxV
        proteins[f] = a
    proteinDict = dict()
    for name, arm in tqdm(armDict.items(), desc= 'Protein loaded for arms'):
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
    for name, a in tqdm(proteins.items(), desc = 'Proteins'):
        a = a.filter(chrom_filter, c='chr'+ chromName)
        a = a.sort()
        # print(name)
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
            allProteins[j][i+1] = score 
            j += 1 
        i += 1
    # if args.normalizeProteins:
        # nop = "_N"
    # else:
        # nop = ""
    # print(allProteins)
    data = pd.DataFrame(allProteins,columns=['start','ctcf', 'rad21', 'smc3',\
                        'H2az', 'H3k4me1', 'H3k4me2', 'H3k4me3','H3k9ac',\
                        'H3k9me3','H3k27ac', 'H3k27me3','H3k36me3',\
                        'H3k79me2', 'H4k20me1'], index=range(len(cuts)))
    return data
    # pickle.dump(data, open(tagCreator(args, "protein") , "wb" ) )


            # # print(str(j+1)+"/"+str(len(cuts)),end='')
            # # print('\r', end='') # use '\r' to go back
            # tmp = a.filter(peak_filter, cut[1], cut[2])
            # print(cut)
            # print(tmp)
            # if args.normalizeProteins:
                # tmp = [(float(x[6]) -minV) / maxV for x in tmp]
            # else:
                # tmp = [float(x[6]) for x in tmp]
            # if len(tmp) == 0:
                # tmp.append(0)
            # score = merge(tmp)
            # allProteins[j][i+1] = score 
            # j += 1 
        # end = time.time()*1000
        # print("Time: %d" % (end - start))
        # i += 1
    # print(args.chrom)
    # if args.normalizeProteins:
        # nop = "_N"
    # else:
        # nop = ""
    # data = pd.DataFrame(allProteins,columns=['start','ctcf', 'rad21', 'smc3', 'H2az', 'H3k4me1', 'H3k4me2', 'H3k4me3',
                              # 'H3k9ac', 'H3k9me3', 'H3k27ac', 'H3k27me3',
                               # 'H3k36me3', 'H3k79me2', 'H4k20me1'],
                               # index=range(len(cuts)))
    # pickle.dump(data, open(tagCreator(args, "protein") , "wb" ) )

