# HiCPrediction
HiCPrediction allows predicting Hi-C matrices using protein levels given by ChIPseq data. 
It is based on random forest regression as proposed by [Zhang, Chasman, Knaack and Roy in 2018](http://dx.doi.org/10.1101/406322). 

HiCPrediction consists of four steps - binning the proteins, creating training/test data sets, training the regression model and prediction, which are described below in more detail. 


## Installation
HiCPrediction requires Python3.6, hicmatrix, hicexplorer, hic2cool, pybedtools and pybigwig.
It is recommended to install everything via conda into an empty environment:

```
$ conda install -c conda-forge -c bioconda -c ska hicmatrix hicexplorer
$ conda install -c conda-forge -c bioconda -c ska hic2cool
$ conda install -c conda-forge -c bioconda -c ska pybedtools pybigwig
$ conda install -c abajorat hicprediction
```
## Scripts

### 1. createBaseFile.py
This script bins the proteins for the given cell line and resolution.
It creates a base file and per-chromosome cooler matrices which are needed as input for the following three steps. 
This step has to be executed for each cell line or resolution with the according protein files.

Options:
 * -mf, --matrixFile PATH  //  Input file with the whole HiC-matrix in cooler format   [required]
 * -ct, --cellType TEXT      Stores cell type for analysis and documentation
                            [required]
 * -r, --resolution TEXT     Stores resolution for analysis and documentation
                            [required]
 * -chs, --chromosomes TEXT  Optional; if set, binning is done only for the
                            chromosomes specified
 * -bf, --baseFile PATH      Name of Base file (.ph5) to be created, where binned proteins are                                  stored for later use. [required]
 * -iod, --internalOutDir   path where internally used matrices will be stored [required]
                            Note that these matrices can be several 100's of MB in size
                            depending on the resolution of the HiC-matrix
 * --help                    Show this message and exit.
  
Arguments: List of protein peaks in narrowPeak format

Examples:
```
$ createBaseFile -mf Gm12878_5kb.cool -bf Gm12878_5kb.ph5 -ct Gm12878 -r 5000 -chs 1,2,3  Gm12878_Rad21.narrowpeak Gm12878_Ctcf.narrowPeak

$ createBaseFile -mf K562_10kb.cool -bf K562_10kb.ph5 -ct K562 -r 10000 K562_chipseqData/*.narrowPeak

```
### 2. createTrainingSet.py
This script creates the datasets that are required later both for training the regression model and for predicting HiC-matrices from given models. An output directory has to be specified as well as the base file that was created in the first step. Unless specified otherwise, data for all chromosomes are converted into training / test datasets with the specific settings chosen by the user.

Options |  Explanation
------|-------
-wo, --windowOperation [avg\|max\|sum] | How the proteins in between two blocks of base pairs should be considered during binning [default: avg]
--ignoreCentromeres BOOL     |   Cut out the centroid arms for training [default: True]
--normalize BOOL             |   normalize protein signal values to a 0-1 range [default: False]
-mo, --mergeOperation [avg\|max] | This parameter defines how the proteins are binned [default: avg]
-chs, --chromosomes TEXT     |   Optional; if set, sets are only computed for the chromosomes specified
-bf, --baseFile PATH         |   Base file with binned proteins for given cell line and resolution, can be created with createBaseFile.  [required]
-dod, --datasetOutputDirectory PATH | Output directory for training set files [required]
-cmf, --centromeresFile PATH |  Text file containing centromer regions. See centromeres.txt in InternalStorage for formatting
-ws, --windowSize INTEGER     |  Maximum distance between two blocks of basepairs [default: 200]
--help                          Show this message and exit.

Example:
```
$ createTrainingSet.py -bf Gm12878_5kb.ph5 -dod Results/Sets/
$ createTrainingSet.py -bf K562_5kb.ph5 -dod Results/Sets/ -chs 1,2,3
```
### 3. training.py
This script builds and trains the regression model. A training set as created above by createTrainingSet.py must be given as well as the output directory for the model.
Optionally, the ChIP-seq read values can be log-converted.


Options:
 * -tdf, --trainDatasetFile PATH   File from which training data is loaded
                                  [required]
 * -co, --conversion [standardLog|none]
                                  Define a conversion function for the read
                                  values  [default: none]
 * -mod, --modelOutputDirectory PATH
                                  Output directory for model files  [required]
 * --help                          Show this message and exit.

Example:
```
$ training.py -tdf Results/Sets/Gm12878_5000_Mavg_Wavg200_Achr1.z -mod Results/Models/

```
### 4. predict.py
This script predicts HiC-matrices based on the chosen model. It requires the basefile created above, a test dataset (created above by createTrainingSet.py) and a model (created above by training.py) and will output the predicted matrix in cooler format.
Optionally a CSV-file can be defined as output for the evaluation metrics.

Options:
 * -bf, --baseFile PATH            Base file with binned proteins for given cell line 
                                    and resolution, can be created with createBaseFile.  [required]
 * -rfp, --resultsFilePath TEXT    File where to store evaluation metrics. If
                                  not set no evaluation is executed
 * -pod, --predictionOutputDirectory PATH
                                  Output directory for prediction files. If
                                  not set, no predicted Hi-C matrices are stored.
                                  Otherwise, the output will be a predicted HiC-matrix in cooler format.
 * -psp, --predictionSetPath PATH  Data set that is to be predicted.
                                  [required]
 * -mfp, --modelFilePath TEXT      Choose model on which to predict  [required]
 * --help                          Show this message and exit.

Example:
```
$ predict.py -bf Gm12878_5kb.ph5 -psp Results/Sets/K562_5000_Mavg_Wavg200_Achr1.z -mfp Results/Models/Gm12878_5000_Mavg_Wavg200_Achr1_Cnone.z -pod Results/Predictions/K562/
```
