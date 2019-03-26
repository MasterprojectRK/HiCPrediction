from hicmatrix import HiCMatrix as hm
from hicexplorer import hicPlotMatrix as hicPlot
import numpy as np
from pprint import pprint
from Autoencoders_Variants import sparse_autoencoder_l1 as SAEL1
from Autoencoders_Variants import data_utils as du

matrix = "../Data/GSE63525_GM12878_insitu_primary_100kb_KR_chr1.cool"
hic_ma = hm.hiCMatrix(matrix)
size = hic_ma.matrix.shape[0]
shape = hic_ma.matrix.shape
print(size)
print(shape)
sliced = hic_ma.matrix[:10,:10]
A = sliced.todense()
B = np.squeeze(np.asarray(A))
num_non_zero = hic_ma.matrix.nnz
sum_elements = hic_ma.matrix.sum() / 2
bin_length = hic_ma.getBinSize()
num_nan_bins = len(hic_ma.nan_bins)
min_non_zero = hic_ma.matrix.data.min()
max_non_zero = hic_ma.matrix.data.max()
chromosomes = list(hic_ma.chrBinBoundaries)
#args = ["--matrix", matrix, "-out", "bla", "--log1p","--clearMaskedBins"]
# fileName = hicPlot.main(args)
# ae = SAEL1.SparseAutoencoderL1()
print(hic_ma.nan_bins)
print(hic_ma.correction_factors)
print(hic_ma.cut_intervals)
print(hic_ma.distance_counts)
print(hic_ma.bin_size)
print(hic_ma.bin_size_homogeneous)
print(hic_ma.bin_size)
print(hic_ma.chrBinBoundaries)
chrom, start, end, extra = zip(*hic_ma.cut_intervals)
median = int(np.median(np.diff(start)))
diff = np.array(end) - np.array(start)
np.set_printoptions(threshold=0)
print(diff)

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
LOG_INTERVAL = 100
SPARSE_REG = 1e-3
TRAIN_SCRATCH = False        # whether to train a model from scratch
BEST_VAL = float('inf')     # record the best val loss
