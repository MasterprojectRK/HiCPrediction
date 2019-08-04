from configurations import *
from tagCreator import createTag

def chrom_filter(feature, c):
        return feature.chrom == c

@click.group()
def cli():
    pass

@click.option('--normalize/--dont-normalize', default=False)
@click.option('--mergeoperation', '-mo', default='avg',\
              type=click.Choice(['avg', 'max']))
@click.option('--cellLine', '-cl', default='Gm12878')
@click.option('resolution', '-r', default=5000)
@click.option('proteinfilepath', '-pfp', default='proteins.h5')
@click.option('chromfilepath', '-cfp', default='chroms.h5')
@click.argument('proteindir')
@cli.command()
def loadAllProteins(proteindir, chromfilepath,proteinfilepath,resolution,\
                    cellline, mergeoperation, normalize):
    column = 6
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
    with h5py.File(chromfilepath, 'a') as chromFile:
        for chrom in tqdm(range(1,23), desc= 'Converting proteins for each chromosome'):
            proteinTag =createTag(resolution, cellline, chrom,\
                                  merge=mergeoperation, norm=normalize)
            chromTag =createTag(resolution, cellline, chrom)

            if chromTag not in chromFile:
                msg = 'The chromosome {} is not loaded yet. Please'\
                        +'update your chromosome file {} using the script'\
                        +'"getChroms"'
                print(msg.format(chromTag, chromfilepath))
                sys.exit()
            cutPath = chromTag + "/bins/start"
            cutsStart = chromFile[cutPath].value
            proteinData = loadProtein(proteins, chrom,cutsStart, column,
                                      mergeoperation) 
            proteinData.to_hdf(proteinfilepath,key=proteinTag, mode='a')


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
