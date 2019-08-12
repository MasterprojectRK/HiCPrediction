#!/usr/bin/python3

from configurations import *
from tagCreator import createProteinTag


@protein_options
@click.argument('proteinFiles', nargs=-1)#, help='Pass all of the protein'\
               # +' files you want to use for the prediction. They should be '+\
               # 'defined for the whole genome or at least, for the chromosomes'\
               # + ' you defined in thee options. Format must be "narrowPeak"')
@click.command()
def loadAllProteins(proteinfiles, basefile,chromosomes,
                   matrixfile,celltype,resolution):
    checkExtension(matrixfile, 'cool', option='h5')
    checkExtension(basefile, 'ph5')
    for fileName in proteinfiles:
        checkExtension(fileName, 'narrowPeak')
    inputName = matrixfile.split(".")[0].split("/")[-1]
    params = dict()
    params['originName'] = inputName
    params['resolution'] = resolution
    params['cellType'] = celltype
    if chromosomes:
        chromosomeList = chromosomes.split(',')
    else:
        chromosomeList = range(1, 23)
    addGenome(matrixfile, basefile, chromosomeList, params)
    for setting in getCombinations():
        with h5py.File(basefile, 'a') as baseFile:
            params['peakColumn'] = setting['peakColumn']
            params['normalize'] = setting['normalize']
            params['mergeOperation'] = setting['mergeOperation']
            proteins = getProteinFiles(proteinfiles, params)
            proteinTag =createProteinTag(params)
            tqdmIter = tqdm(chromosomeList, desc= 'Iterate chromosomes')
            for chromosome in tqdmIter:
                chromTag = "chr" +str(chromosome)
                cutPath = chromTag + "/bins/start"
                cutsStart = baseFile[cutPath].value
                proteinData = loadProtein(proteins, chromosome,cutsStart,params)
                proteinChromTag = proteinTag + "_" + chromTag
                store = pd.HDFStore(basefile)
                store.put(proteinChromTag, proteinData)
                store.get_storer(proteinChromTag).attrs.metadata = params
                store.close()
    print("\n")

def getProteinFiles(proteinFiles, params):
    proteins = dict()
    for path in proteinFiles:
        a = pybedtools.BedTool(path)
        b = a.to_dataframe()
        c = b.iloc[:,params['peakColumn']]
        minV = min(c)
        maxV = max(c) - minV
        if maxV == 0:
            maxV = 1.0
        if params['normalize']:
            for row in a:
                row[params['peakColumn']] = str((float(\
                    row[params['peakColumn']]) - minV) / maxV)
        proteins[path] = a
    return proteins

def loadProtein(proteins, chromName, cutsStart, params):
        i = 0
        allProteins = []
        if params['mergeOperation'] == 'avg':
            merge = np.mean
        elif params['mergeOperation'] == 'max':
            merge = np.max
        for cut in cutsStart:
            allProteins.append(np.zeros(len(proteins) + 1))
            allProteins[i][0] = cut
            i += 1
        i = 0
        columns = ['start']
        for name, a in tqdm(proteins.items(), desc = 'Proteins converting'):
            columns.append(str(i))
            a = a.filter(chrom_filter, c='chr'+ str(chromName))
            a = a.sort()
            values = dict()
            for feature in a:
                peak = feature.start + int(feature[9])
                pos = bisect.bisect_right(cutsStart, peak)
                if pos in values:
                    values[pos].append(float(feature[params['peakColumn']]))
                else:
                    values[pos] = [float(feature[params['peakColumn']])]
            j = 0
            for key, val in values.items():
                score = merge(val)
                allProteins[key - 1][i+1] = score
            i += 1
        data = pd.DataFrame(allProteins,columns=columns, index=range(len(cutsStart)))
        return data

def addGenome(matrixFile, chromosomeOutputFile, chromosomeList, params):
    with h5py.File(chromosomeOutputFile, 'a') as f:
        for i in tqdm(chromosomeList,desc='Converting chromosomes'):
            tag = "chr" + str(i)
            if not tag in f:
                sub2 = "hicAdjustMatrix -m "+matrixFile +" --action keep --chromosomes " +\
                str(i)+" -o tmp/chrom"+str(i)+".h5"
                subprocess.call(sub2,shell=True)
                with h5py.File('tmp/chrom'+str(i)+'.h5', 'r') as m:
                    m.copy('/', f,name=tag)
    for f in os.listdir("tmp/"):
        os.remove('tmp/'+f)

def chrom_filter(feature, c):
        return feature.chrom == c

if __name__ == '__main__':
    loadAllProteins()

