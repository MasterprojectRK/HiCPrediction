import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
#from hicmatrix import HiCMatrix as hm
import cooler
import pandas as pd
import sys
import click
from scipy import sparse
import numpy as np

@click.option('--outfile','-o',type=click.Path(writable=True),help="name/path of cooler file to write", required=True)
@click.option('--peakpos','-pp',default=5000000,type=click.IntRange(min=0),help="position of peak", required=False)
@click.option('--peakwidth','-pw',default=2000000,type=click.IntRange(min=100000,max=3000000),help="width of peak",required=False)
@click.option('--length','-l',default=10000000,type=click.IntRange(min=5000000,max=20000000),help="length of train/test matrix in bp",required=False)
@click.option('--resolution','-r',default=20000,type=click.Choice(choices=[5000,10000,20000]),help="output matrix resolution")
@click.option('--count','-c',default=100,type=click.IntRange(min=1,max=1000),help="count value for peak")
@click.option('--chromosome','-chr',default='chr1')
@click.command()
def createFakeMatrices(outfile,peakpos,peakwidth,length,resolution,count,chromosome):
    errorMsg=""
    if not outfile.endswith('.cool'):
        errorMsg += "Matrix output file must be in cooler format. Aborting\n"
    if peakwidth > length/2:
        errorMsg += "peak width must not be more than half the peak length\n"
    if peakpos - peakwidth/2 < 0 or peakpos + peakwidth/2 > length:
        errorMsg += "Peak is not fully inside the range (0...length). Reduce peak width or adjust peak position\n"
    if errorMsg != "":
        sys.exit(errorMsg)
    
    adjustedLength = length - length%resolution
    binStartList = list(range(0,adjustedLength,resolution))
    binEndList = list(range(resolution,adjustedLength,resolution))
    binEndList.append(adjustedLength)
    if len(binStartList) != len(binEndList):
        errorMsg = "bug while creating bins. Start and end bin lists not equally long"
        sys.exit(errorMsg)
    bins = pd.DataFrame(columns=['chrom','start','end'])
    bins['start'] = binStartList
    bins['end'] = binEndList
    bins['chrom'] = chromosome

    bin1List = []
    bin2List = []
    for bin1Id in range(len(binStartList)):
        for bin2Id in range(len(binStartList)):
            bin1List.append(bin1Id)
            bin2List.append(bin2Id)
    
    pixels = pd.DataFrame(columns=['bin1_id','bin2_id','count'])
    pixels['bin1_id'] = bin1List
    pixels['bin2_id'] = bin2List
    pixels['count'] = 0
   
    adjustedPeakWidth = peakwidth - peakwidth%resolution
    peakStartBin = int((peakpos-adjustedPeakWidth/2)/resolution)
    peakEndBin = peakStartBin + int(adjustedPeakWidth/resolution)
    m1 = pixels['bin1_id'] >= peakStartBin
    m2 = pixels['bin1_id'] < peakEndBin
    m3 = pixels['bin2_id'] >= peakStartBin
    m4 = pixels['bin2_id'] < peakEndBin
    mask = m1 & m2 & m3 & m4
    pixels.loc[mask,'count'] = count
    pixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)
    
    #assert that the resulting matrix is symmetric
    matIdx = (list(pixels['bin1_id']), list(pixels['bin2_id']))
    data = list(pixels['count'])
    mtrx = sparse.csr_matrix((data,matIdx)).todense()
    symmetric = np.allclose(mtrx, mtrx.T, rtol=1e-20, atol=1e-20)
    if not symmetric:
        errorMsg = 'bug: resulting matrix should be symmetric, but is not'
        sys.exit(errorMsg)


    cooler.create_cooler(outfile, bins=bins, pixels=pixels, triucheck=False, symmetric_upper=False)


if __name__ == '__main__':
    createFakeMatrices()