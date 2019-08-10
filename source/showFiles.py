from hiCOperations import *




@click.argument('filename', type=click.Path(exists=True))
@cli.command()
def showH5(filename):
    # with h5py.File(filename, 'r') as f:
        for i in range(1,23):
            proteins = pd.read_hdf(fileame,key='chr'+str(i), mode='r')
            print(proteins.shape)


def printName(name):
    print(f[name].shape)

if __name__ == '__main__':
    showH5()
