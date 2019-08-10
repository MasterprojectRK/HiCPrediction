#!/usr/bin/python3

from configurations import *
from tagCreator import createProteinTag


@protein_options
@standard_options
@click.argument('proteinInputFiles', nargs=-1)
@cli.command()
def loadAllProteins(proteininputfiles, experimentoutputdirectory,chromosomes,
                   matrixfile,normalize, peakcolumn, mergeoperation):
    if chromosomes:
        chromosomeList = chromosomes.split(',')
    else:
        chromosomeList = range(1, 23)
    chromosomeFileName = experimentoutputdirectory + '/chromosomes.ch5'
    if not os.path.isdir(experimentoutputdirectory):
        os.mkdir(experimentoutputdirectory)
    if not os.path.isdir(experimentoutputdirectory + "/Proteins"):
        os.mkdir(experimentoutputdirectory + "/Proteins")
    if not os.path.isfile(chromosomeFileName):
        addGenome(matrixfile, chromosomeFileName, chromosomeList)
    inputName = matrixfile.split(".")[0].split("/")[-1]
    params = dict()
    params['peakColumn'] = peakcolumn
    params['normalize'] = normalize
    params['mergeOperation'] = mergeoperation
    with h5py.File(chromosomeFileName, 'a') as chromFile:
        proteins = getProteinFiles(proteininputfiles, params)
        proteinTag =createProteinTag(params)
        for chromosome in tqdm(chromosomeList, desc= 'Converting proteins for each chromosome'):
            chromTag = "chr" +str(chromosome)
            cutPath = chromTag + "/bins/start"
            cutsStart = chromFile[cutPath].value
            proteinData = loadProtein(proteins, chromosome,cutsStart,params)
            fileName = experimentoutputdirectory + "/Proteins/" +proteinTag +".ph5"
            proteinData.to_hdf(fileName,key=chromTag, mode='a')

def getProteinFiles(proteinInputFiles, params):
    proteins = dict()
    for path in proteinInputFiles:
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

def addGenome(matrixFile, chromosomeOutputFile, chromosomeList):
    with h5py.File(chromosomeOutputFile, 'a') as f:
        for i in tqdm(chromosomeList,desc='Converting chromosomes'):
            tag = "chr" + str(i)
            if not tag in f:
                sub2 = "hicAdjustMatrix -m "+matrixFile +" --action keep --chromosomes " +\
                str(i)+" -o tmp/chrom"+str(i)+".h5"
                subprocess.call(sub2,shell=True)
                with h5py.File('tmp/chrom'+str(i)+'.h5', 'r') as m:
                    m.copy('/', f,name=tag)
    for f in os.listdi("tmp/"):
        os.remove(f)

def chrom_filter(feature, c):
        return feature.chrom == c

if __name__ == '__main__':
    loadAllProteins()
