from hiCOperations import *
from tagCreator import createTag


def cli():
    pass

def addGenome(matrixfile, experimentoutputdirectory):
    inputFormat = matrixfile.split(".")[-1]
    inputName = matrixfile.split(".")[0].split("/")[-1]
    with h5py.File(experimentoutputdirectory + "/" + inputName \
                   + '.chromh5', 'a') as f:
        for i in tqdm(range(1,23)):
            tag = "chr" + str(i)
            if not tag in f:
                sub2 = "hicAdjustMatrix -m "+matrixfile +" --action keep --chromosomes " +\
                str(i)+" -o tmp/chrom"+str(i)+".h5"
                subprocess.call(sub2,shell=True)
                with h5py.File('tmp/chrom'+str(i)+'.h5', 'r') as m:
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

if __name__ == '__main__':
    cli()

