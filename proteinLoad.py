from configurations import *
from tagCreator import createTag

def chrom_filter(feature, c):
        return feature.chrom == c

@click.group()
def cli():
    pass

@standard_options
@protein_options
@proteinfile_options
@chromfile_options
@click.argument('proteindir')
@cli.command()
def loadAllProteins(proteindir, chromfilepath,proteinfilepath,resolution,\
                    cellline, mergeoperation, normalize, proteincolumn):

    proteins = getProteinFiles(proteincolumn, cellline, proteindir, normalize)

    with h5py.File(chromfilepath, 'a') as chromFile:
        for chrom in tqdm(range(1,23), desc= 'Converting proteins for each chromosome'):
            saveData(resolution, cellline, chrom, mergeoperation, normalize,
            proteinfilepath, chromFile, chromfilepath, proteins, proteincolumn)


@standard_options
@protein_options
@click.option('--chromfilepath', '-cfp', default='Data/chroms.h5',\
                 type=click.Path(exists=True), show_default=True)
@click.option('--proteinfilepath', '-pfp', default='Data/proteins.h5',
              type=click.Path(exists=True), show_default=True)
@click.argument('proteindir')
@click.argument('chromosome')
@cli.command()
def loadProteinsForOneChromosome(chromosome, proteindir, chromfilepath,\
        proteinfilepath,resolution,cellline, mergeoperation,\
                                 normalize,proteincolumn):
    proteins = getProteinFiles(proteincolumn, cellline, proteindir, normalize)

    with h5py.File(chromfilepath, 'a') as chromFile:
        saveData(resolution, cellline, chromosome, mergeoperation, normalize,
                proteinfilepath, chromFile, chromfilepath, proteins,\
                 proteincolumn)



def saveData(resolution, cellline, chrom, mergeoperation, normalize,\
        proteinfilepath, chromFile, chromfilepath, proteins, pc):
    proteinTag =createTag(resolution, cellline, chrom,\
                          merge=mergeoperation, norm=normalize, pc=pc)
    chromTag =createTag(resolution, cellline, chrom)

    if chromTag not in chromFile:
        msg = 'The chromosome {} is not loaded yet. Please'\
                +'update your chromosome file {} using the script'\
                +'"getChroms"'
        print(msg.format(chromTag, chromfilepath))
        sys.exit()
    cutPath = chromTag + "/bins/start"
    cutsStart = chromFile[cutPath].value
    proteinData = loadProtein(proteins, chrom,cutsStart, pc,
                              mergeoperation) 
    proteinData.to_hdf(proteinfilepath,key=proteinTag, mode='a')


def getProteinFiles(column, cellline, proteindir, normalize):
    proteins = dict()
    for f in os.listdir(proteindir):
        if f.startswith(cellline):
            path  = proteindir+f
            a = pybedtools.BedTool(path)
            b = a.to_dataframe()
            c = b.iloc[:,column]
            minV = min(c)
            maxV = max(c) - minV
            if maxV == 0:
                maxV = 1
            if normalize:
                for row in a:
                    row[column] = (float(x[column]) - minV) / maxV
            proteins[f] = a
    return proteins

def loadProtein(proteins, chromName, cutsStart, column, mergeOperation):
        i = 0
        allProteins = []
        if mergeOperation == 'avg':
            merge = np.mean
        elif mergeOperation == 'max':
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
                    values[pos].append(float(feature[column]))
                else:
                    values[pos] = [float(feature[column])]
            j = 0
            for key, val in values.items():
                score = merge(val)
                allProteins[key - 1][i+1] = score
            i += 1
        data = pd.DataFrame(allProteins,columns=columns, index=range(len(cutsStart)))
        return data

if __name__ == '__main__':
    cli()
