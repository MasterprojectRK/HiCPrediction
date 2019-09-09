# HiCPrediction
HiCPrediction enables the training of Random Forest Regressor models for Hi-C matrices by using protein levels. The models can then be used to predict Hi-C matrices of other chromosomes and cell lines. The current version requires an existing Hi-C matrix, or at least the binning intervals for any prediction. The framework has 2 data preparation scripts, one training scrip and one prediction script. All of them have to be executed consecutively in order to work. 
## Installment
Unordered lists can be started using the toolbar or by typing `* `, `- `, or `+ `. Ordered lists can be started by typing `1. `.

## Scripts

### 1. createBaseFile'
This script creates bins the proteins for the given cell line and creates a base file that is needed for the consecutive steps. This step has to be executed for each cell line or resolution with the according protein files.
The following arguments can be passed
Options:
 * -mf, --matrixFile PATH  //  Input file with the whole HiC-matrix   [required]
 * -ct, --cellType TEXT      Store cell type for analysis and documentation
                            [required]
 * -r, --resolution TEXT     Store resolution for analys and documentation
                            [required]
 * -chs, --chromosomes TEXT  If set, sets are only calculated for these
                            chromosomes instead of all
 * -bf, --baseFile PATH      Base file where to store proteins and chromosomes
                            for later use.  [required]
 * --help                    Show this message and exit.
  
Arguments:
                            List of proteinfiles (narrowpeak format)


Example:
```
$ createBaseFile -mf hic.cool -bf basefile.ph5 -ct Gm12878 -r  -chs 1,2,3 5000 Gm12878_Rad21.narrowpeak Gm12878_Ctcf.narrowpeak
```
### 2. createTrainingSet.py'
This script creates bins the proteins for the given cell line and creates a base file that is needed for the consecutive steps. This step has to be executed for each cell line or resolution with the according protein files.
The following arguments can be passed

 * -wo, --windowOperation [avg|max|sum] 
                                  How should the proteins in between two base
                                  pairs be summed up  [default: avg]
 * --ignoreCentromeres TEXT        Cut out the centroid arms for training
                                  [default: True]
*  --normalize TEXT                Should the proteins be normalized to a 0-1
                                  range  [default: False]
 * -mo, --mergeOperation [avg|max]
                                  This parameter defines how the proteins are
                                  binned  [default: avg]
 * -chs, --chromosomes TEXT        If set, sets are only calculated for these
                                  chromosomes instead of all
 * -bf, --baseFile PATH            Base file where to store proteins and
                                  chromosomes for later use.  [required]
 * -dod, --datasetOutputDirectory PATH
                                  Output directory for training set files
                                  [required]
 * -cmf, --centromeresFile PATH
 * -ws, --windowSize INTEGER       Maximum distance between two basepairs
                                  [default: 200]
Example:
```
$ createBaseFile -mf hic.cool -bf basefile.ph5 -ct Gm12878 -r 5000 Gm12878_Rad21.narrowpeak Gm12878_Ctcf.narrowpeak
```
