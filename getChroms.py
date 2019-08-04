from hiCOperations import *


@click.group()
def cli():
    pass

@click.option('--cellLine', '-cl', default='Gm12878')
@click.option('resolution', '-r', default=5000)
@click.option('chromfilepath', '-cfp', default='chroms.h5')
@click.argument('sourceFile')
@cli.command()
def addGenome(sourcefile,chromfilepath, resolution, cellline):
    inputFormat = sourcefile.split(".")[-1]
    if inputFormat != "cool" and inputFormat != "h5":

        sub1 = "hicConvertFormat -m "+ sourcefile+" --inputFormat "+ inputFormat +\
            " --outputFormat h5 -o tmp/matrix.h5 --resolutions " +resolution
        subprocess.call(sub1,shell=True)
        tmpFile = "tmp/matrix.h5"
    else:
        tmpFile = sourcefile
    with h5py.File(chromfilepath, 'a') as f:
        for i in tqdm(range(1,23)):
            tag = createTag(resolution, cellline, chrom=i)
            if not tag in f:
                sub2 = "hicAdjustMatrix -m "+tmpFile +" --action keep --chromosomes " +\
                str(i)+" -o tmp/chrom"+str(i)+".h5"
                subprocess.call(sub2,shell=True)
                with h5py.File('tmp/chrom'+str(i)+'.h5', 'r') as m:
                    m.copy('/', f,name=tag)
    for f in os.files("tmp/"):
        os.remove(f)


@click.option('chromfilepath', '-cfp', default='chroms.h5')
@click.option('--cellLine', '-cl', default='Gm12878')
@click.option('resolution', '-r', default=5000)
@click.argument('chromNumber')
@click.argument('sourceFile')
@cli.command()
def addChrom(sourceFile,chromnumber, resolution, cellline, chromfilepath):
    inputFormat = sourcefile.split(".")[-1]
    tag = createTag(resolution, cellline, chrom=chromnumber)
    with h5py.File(chromfilepath, 'a') as f:
        sub1 = "hicConvertFormat -m "+ sourcefile+" --inputFormat "+ inputFormat +\
            " --outputFormat h5 -o tmp/chrom"+str(chromNumber)+".h5"
        subprocess.call(sub1,shell=True)
        with h5py.File('tmp/chrom'+str(chromNumber)+'.h5','r') as m:
            del f[tag]
            m.copy('/', f,name=tag)
    for f in os.files("tmp/"):
        os.remove(f)

@cli.command()
def showChroms():
    with h5py.File('chroms.h5', 'a') as f:
        f.visit(printName)
        # df = pd.read_hdf('proteins.h5', key = '5000/Gm12878/chr9')
        # print(df.head(100))


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

