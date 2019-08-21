#!/usr/bin/env python3

from configurations import *
from tagCreator import createProteinTag

""" Module responsible for the binning of proteins and the cutting of
    the genome HiC matrix into the chromosomes for easier access.
    Creates base file that stores binned proteins and paths to the 
    chromosome HiC matrices.
"""

@protein_options
@click.argument('proteinFiles', nargs=-1)#, help='Pass all of the protein'\
               # +' files you want to use for the prediction. They should be '+\
               # 'defined for the whole genome or at least, for the chromosomes'\
               # + ' you defined in thee options. Format must be "narrowPeak"')
@click.command()
def loadAllProteins(proteinfiles, basefile,chromosomes,
                   matrixfile,celltype,resolution):
    """
    Main function that is called with the desired path to the base file, a list
    of chromosomes that are to be included, the source HiC file (whole genome),
    the cell type and the resolution of said HiC matrix and most importantly
    the paths to the protein files that should be used
    Attributes:
        proteinfiles -- list of paths of the protein files to be processed
        basefile --  output path for base file
        chromosomes -- list of chromosomes to be processed
        matrixfile -- path to input HiC matrix
        celltype -- cell line of the input matrix
        resolution -- resolution of the input matrix
    """
    ### checking extensions of files
    checkExtension(matrixfile, 'cool')
    checkExtension(basefile, 'ph5')
    for fileName in proteinfiles:
        checkExtension(fileName, 'narrowPeak')
    ### cutting path to get source file name for storage
    inputName = matrixfile.split(".")[0].split("/")[-1]
    ### creation of parameter set
    params = dict()
    params['originName'] = inputName
    params['resolution'] = resolution
    params['cellType'] = celltype
    ### conversion of desired chromosomes to list
    if chromosomes:
        chromosomeList = chromosomes.split(',')
    else:
        chromosomeList = range(1, 23)
    outDirectory = "InternalStorage/"
    ### call of function responsible for cutting and storing the chromosomes
    chromosomeDict = addGenome(matrixfile, basefile, chromosomeList,
                               outDirectory)
    with h5py.File(basefile, 'a') as baseFile:
        ### iterate over possible combinations for protein settings
        for setting in getCombinations():
            params['peakColumn'] = setting['peakColumn']
            params['normalize'] = setting['normalize']
            params['mergeOperation'] = setting['mergeOperation']
            ### get protein files with given paths for each setting
            proteins = getProteinFiles(proteinfiles, params)
            proteinTag =createProteinTag(params)
            ### literate over all chromosomes in list 
            for chromosome in tqdm(chromosomeDict.keys(), desc= 'Iterate chromosomes'):
                ### get bin boundaries of HiC matrix
                cuts = chromosomeDict[chromosome].cut_intervals
                cutsStart = [cut[1] for cut in cuts]
                ### call function that actually bins the proteins
                proteinData = loadProtein(proteins, chromosome,cutsStart,params)
                proteinChromTag = proteinTag + "_" + chromosome
                ### store binned proteins in base file
                store = pd.HDFStore(basefile)
                store.put(proteinChromTag, proteinData)
                store.get_storer(proteinChromTag).attrs.metadata = params
                store.close()
    print("\n")

def getProteinFiles(proteinFiles, params):
    """ function responsible of loading the protein files from the paths that
    were given
    Attributes:
        proteinfiles -- list of paths of the protein files to be processed
        params -- dictionary with parameters
    """
    proteins = dict()
    for path in proteinFiles:
        a = pybedtools.BedTool(path)
        ### compute min and max for the normalization
        b = a.to_dataframe()
        c = b.iloc[:,params['peakColumn']]
        minV = min(c)
        maxV = max(c) - minV
        if maxV == 0:
            maxV = 1.0
        items = []
        if params['normalize']:
            ### normalize rows if demanded
            for row in a:
                row[params['peakColumn']] = str((float(\
                    row[params['peakColumn']]) - minV) / maxV)
                items.append(row)
            a = pybedtools.BedTool(items)
        proteins[path] = a
    return proteins

def loadProtein(proteins, chromName, cutsStart, params):
    """
    bin protein data and store for the given chromosome
    Attributes:
        protein -- list of loaded bed files for each protein
        chromName --  index of specific chromosome
        cutsStart -- starting positions of bins
        params -- dictionary with parameters
    """

    i = 0
    allProteins = []
    ### create binning function as demanded
    if params['mergeOperation'] == 'avg':
        merge = np.mean
    elif params['mergeOperation'] == 'max':
        merge = np.max
    ### create data structure
    for cut in cutsStart:
        allProteins.append(np.zeros(len(proteins) + 1))
        allProteins[i][0] = cut
        i += 1
    i = 0
    columns = ['start']
    ### iterate proteins 
    for name, a in tqdm(proteins.items(), desc = 'Proteins converting'):
        columns.append(str(i))
        ### filter for specific chromosome and sort
        a = a.filter(chrom_filter, c=str(chromName))
        a = a.sort()
        values = dict()
        for feature in a:
            peak = feature.start + int(feature[9])
            ### get bin index of peak
            pos = bisect.bisect_right(cutsStart, peak)
            if pos in values:
                values[pos].append(float(feature[params['peakColumn']]))
            else:
                values[pos] = [float(feature[params['peakColumn']])]
        j = 0
        for key, val in values.items():
            ### bin proteins for each bin
            score = merge(val)
            allProteins[key - 1][i+1] = score
        i += 1
    data = pd.DataFrame(allProteins,columns=columns, index=range(len(cutsStart)))
    return data

def addGenome(matrixFile, baseFilePath, chromosomeList, outDirectory):
    """
    function that cuts genome HiC matrix into chromosomes and stores them
    internally
    Attributes:
        matrixfile -- path to input HiC matrix
        baseFilePath --  output path for base file
        chromosomeList -- list of chromosomes to be processed
        outDirectory --  output path for chromosomes
    """

    chromosomeDict = {}
    with h5py.File(baseFilePath, 'a') as baseFile:
        for i in tqdm(chromosomeList,desc='Converting chromosomes'):
            tag = outDirectory + matrixFile.split("/")[-1].split(".")[0]\
                    +"_chr" + str(i) +".cool"
            chromTag = "chr" + str(i)
            ### create process to cut chromosomes
            sub2 = "hicAdjustMatrix -m "+matrixFile +" --action keep" \
                        +" --chromosomes " + str(i) + " -o " + tag
            ### execute call if necessary
            if not chromTag in baseFile:
                subprocess.call(sub2,shell=True)
                baseFile[chromTag] = tag
            elif not os.path.isfile(baseFile[chromTag].value):
                subprocess.call(sub2,shell=True)
            ### store HiC matrix
            chromosomeDict[chromTag] =hm.hiCMatrix(tag)
    return chromosomeDict

def chrom_filter(feature, c):
        return feature.chrom == c

if __name__ == '__main__':
    loadAllProteins()

