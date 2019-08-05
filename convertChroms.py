from hiCOperations import *


@click.group()
def cli():
    pass

# @standard_options
# @chromfile_options
@click.option('--chromosomesOutputFile', '-cof', type=click.Path(exists=True))
@click.argument('matrixFile')
@cli.command()
def addGenome(matrixFile, chromosomesOutputFile):
    inputFormat = matrixFile.split(".")[-1]
    with h5py.File(chromosomeOutputFile, 'a') as f:
        for i in tqdm(range(1,23)):
            tag = createTag(matrixFile, chrom=i)
            if not tag in f:
                sub2 = "hicAdjustMatrix -m "+matrixFile +" --action keep --chromosomes " +\
                str(i)+" -o tmp/chrom"+str(i)+".h5"
                subprocess.call(sub2,shell=True)
                with h5py.File('tmp/chrom'+str(i)+'.h5', 'r') as m:
                    m.copy('/', f,name=tag)
    for f in os.files("tmp/"):
        os.remove(f)


@standard_options
@click.option('--chromtargetfilepath', '-ctp', default='Data/chroms.h5',\
                 type=click.Path(exists=True), show_default=True)
@click.option('--genomeFilePath','-gfp', default=None, cls=Mutex,\
              not_required_if = ['chromsourcefilepath'])
@click.option('--chromsourcefilepath','-csp', default=None, cls=Mutex,\
              not_required_if = ['genomefilepath'])
@click.argument('chromNumber')
@cli.command()
def addChrom(genomefilepath,chromnumber,chromsourcefilepath, resolution,\
             cellline, chromtargetfilepath):
    if genomefilepath:
        inputFormat = genomefile.split(".")[-1]
        with h5py.File(chromtargetfilepath, 'a') as f:
            sub1 = "hicConvertFormat -m "+ genomefile+" --inputFormat "+ inputFormat +\
                " --outputFormat h5 -o tmp/chrom"+str(chromnumber)+".h5"
            subprocess.call(sub1,shell=True)
            fileName = 'tmp/chrom'+str(chromnumber)+'.h5'
    elif chromsourcefilepath:
        fileName = chromsourcefilepath
    else: 
                msg = 'You have to set either(xor) a whole genome file to load from '+\
                'or a single chromosome file.'
                print(msg)
                sys.exit()
    tag = createTag(resolution, cellline, chrom=chromnumber)
    with h5py.File(fileName ,'r') as m:
        del f[tag]
        m.copy('/', f,name=tag)
    for f in os.files("tmp/"):
        os.remove(f)


@click.argument('filename', type=click.Path(exists=True))
@cli.command()
def showH5(filename):
    with h5py.File(filename, 'r') as f:
        f.visit(printName)


def printName(name):
    print(name)


def divideIntoArms(args, chromDict):
    armDict = dict()
    for name, chrom in tqdm(chromDict.items()):
        if(name in ['13','14','15','22']):
            armDict[name+"_A"] = chrom
        else:
            f=open("Data/Centromeres/centromeres.txt", "r")
            fl =f.readlines()
            elems = None
            for x in fl:
                elems = x.split("\t")
                if elems[1] == "chr"+name:
                    continue
            start = int(elems[2])
            end = int(elems[3])
            cuts = chrom.cut_intervals
            i = 0
            cuts1 = []
            cuts2 = []
            firstIndex = 0
            for cut in cuts:
                if cut[2] < start:
                    cuts1.append(cut)
                    lastIndex = i + 1
                elif cut[1] > end:
                    cuts2.append(cut)
                else:
                    firstIndex = i + 1
                i += 1
            if firstIndex == 0:
                firstIndex = lastIndex
            ma_a = chrom
            ma_b = deepcopy(chrom)
            m1 = ma_a.matrix.todense()
            m2 = ma_b.matrix.todense()
            m1 = m1[:lastIndex,:lastIndex]
            new = sparse.csr_matrix(m1)
            ma_a.setMatrix(new, cuts1)
            m2 = m2[firstIndex:,firstIndex:]
            new = sparse.csr_matrix(m2)
            ma_b.setMatrix(new, cuts2)
            armDict[name+"_A"] = ma_a
            armDict[name+"_B"] = ma_b
    return armDict

if __name__ == '__main__':
    cli()

