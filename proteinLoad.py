from configurations import *
from tagCreator import createProteinTag, createChromTag
from convertChroms import addGenome

def chrom_filter(feature, c):
        return feature.chrom == c

@click.group()
def cli():
    pass
@click.argument('proteinInputFiles', nargs=-1)
@protein_options
@click.option('--experimentOutputDirectory','-eod', required=True,\
              help='Outputfile', type=click.Path(exists=True))
@click.option('--matrixFile', '-mf', required=True,help='Input file',\
              type=click.Path(exists=True))
@cli.command()
def loadAllProteins(proteininputfiles, experimentoutputdirectory,
                   matrixfile,normalize, peakcolumn, mergeoperation):
    addGenome(matrixfile, experimentoutputdirectory)
    inputName = matrixfile.split(".")[0].split("/")[-1]
    chromosomeFileName = experimentoutputdirectory + '/' + inputName + 'chromh5'
    params = dict()
    params['peakColumn'] = peakcolumn
    params['normalize'] = normalize
    params['mergeOperation'] = mergeoperation
    with h5py.File(chromosomeFileName, 'a') as chromFile:
        proteins = getProteinFiles(proteininputfiles, params)
        proteinTag =createProteinTag(inputName ,params)
        for chromosome in tqdm(range(1,23), desc= 'Converting proteins for each chromosome'):
            chromTag = "chr" +str(chromsome)
            cutPath = chromTag + "/bins/start"
            cutsStart = chromFile[cutPath].value
            proteinData = loadProtein(proteins, chromosome,cutsStart,params)
            fileName = experimentoutputdirectory + "/" +proteinTag + ".proteinh5"
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
                row[params['peakColumn']] = str((float(row[params['peakColumn']]) - minV) / maxV)
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
            allProteins.append(np.zeros(15))
            allProteins[i][0] = cut
            i += 1
        i = 0
        columns = ['start']
        for name, a in tqdm(proteins.items(), desc = 'Proteins converting'):
            protein = name.split(".")[0].split("_")[-1]
            columns.append(protein)
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

if __name__ == '__main__':
    cli()
