# HiCPrediction
HiCPrediction allows predicting unknown Hi-C matrices using known
Hi-C matrices and protein levels given by ChIPseq data. 
It is based on random forest regression as proposed by [Zhang, Chasman, Knaack and Roy in 2018](http://dx.doi.org/10.1101/406322). 

HiCPrediction consists of four steps
* binning the proteins according to training matrix resolution, 
* creating training and test data sets, 
* training a regression model on given HiC-matrices and ChIP-seq data,
* prediction of unknown HiC-matrices using the regression model and ChIP-seq data

These four steps are described below in more detail. 


## Installation
HiCPrediction requires Python >= 3.6, click, cooler, graphviz,
h5py, hicexplorer, hicmatrix, hyperopt, joblib, matplotlib, numpy, pandas,
pybedtools, pyBigWig, pydot, scikit-learn, scipy and tqdm.

It is recommended to install hicprediction using a local conda package.
To do so, first download the meta.yaml provided in our github repository
and change into the download directory. 

```
# first, download and install conda, if not already present

# create a new conda environment (optional, but recommended)
$ conda create -n hicprediction
$ conda activate hicprediction

# go to the directory where the meta.yaml has been stored
# and build a local conda package.
# The command will print out the 
# location where the package has been placed
$ conda build .

# install the local package
$ conda install hicprediction --use-local
```


## Usage

### 1. createBaseFile
This script aggregates given ChIP-seq data for a certain cell line and resolution into bins of user-defined size.
It creates a so-called basefile and also per-chromosome cooler matrices, if provided with a Hi-C matrix. 
Basefiles are required both for building training and test datasets (see below). Basefiles can span several or all numerical human chromosomes, but separate files have to be created for each cell line and each resolution, using the appropriate protein files.

Options:
 * -bf, --baseFile PATH [required]
    
    Name of Base file to be created, should have .ph5 file extension 
 
 * -ct, --cellType TEXT [required]      
 
    User-defined cell type for analysis and documentation.
    This is required for naming output files, computing statistics etc.
                            
 * -r, --resolution INT [required]    
 
    Resolution (= bin size) in basepairs, must match resolution of Hi-C matrix (if provided)
                            
 * -chs, --chromosomes LIST OF TEXT [optional]
 
    Optional; if set, binning is done only for the
chromosomes specified, otherwise for all numerical ones. Must be a comma separated list without 'chr', e.g. '1' or '1,3,21'
 * -csf PATH [required]  

    chromosome size file, see e.g. hg19.chrom.sizes for formatting. Required for binning.

 * -mf, --matrixFile PATH [optional]  
 
    Hi-C matrix in cooler format, required if basefile is to be used for training later in the process (see below) 

 * -iod, --internalOutDir PATH [optional]  
 
    Path where internally used matrices will be stored. 
Note that these matrices can be several 100's of MB in size
depending on the resolution of the Hi-C matrix. Must be used if matrixFile is specified. The matrices can be moved to another folder afterwards, but must not be renamed, otherwise they will not be found during the (training-)dataset creation step in 'createTrainingSet'.


 * --help 

   show help message and exit
  
* Positional arguments: LIST OF PATH [required] 

    List of ChIP-seq input files in narrowPeak, broadPeak or bigwig format. All formats can be used simultanously. Note that the runtime is in O(n log(n)) for n = number of features = 3x number of ChIP-seq input files. 
    Specifying folders with protein data is possible using wildcards,
    e.g. /path/to/bigwig/files/*.bigwig 
    The proteins must be the same for training / test datasets and they must be in the same order, see examples.

Returns:
 * basefile with user-defined name and .ph5 file extension (see option -bf)

### 2. createTrainingSet
This script creates the datasets that are required later both for training the regression model and for predicting Hi-C matrices from given models. 

* -ws, --windowSize INT [optional]
    
    Maximum bin distance which will be considered for training and prediction. Default: 200; corresponds to 1.000.000 bp at 5kb resolution.

* -dod, --datasetOutputDirectory PATH [required]

    Output directory for training set files

* -mo, --mergeOperation {avg\|max} [optional]

    Proteins can be aggregated by taking the mean (avg) or the max signal value within each bin

* --divideProteinsByMean BOOL [optional]

    Divide each protein signal by its mean. Default: False

* --normalizeProteins BOOL [optional]

    Scale protein signal values to value range [0...normSignalValue].
    Default: False

* --normSignalValue FLOAT [optional]

    Max. protein signal value after scaling. Default: 1000.0

* --normSignalThreshold FLOAT [optional]

    Set all values smaller than 'normSignalThreshold' to 0 after signal value scaling. Default: 0.1

* --normalizeReadCounts BOOL [optional]

    Scale Hi-C matrix read counts to value range [0...normCountValue]. Only useful when the basefile has been created with the -matrixFile option set.
    Default: False

* --normCountValue FLOAT [optional]

    Max. read count value after scaling. 
    Only useful when the basefile has been created with the -matrixFile option set. Default: 1000.0

* --normCountThreshold FLOAT [optional]

    Set all values smaller than 'normCountThreshold' to 0 after read count scaling.  
    Only useful when the basefile has been created with the -matrixFile option set. Default: 0.1.
   
* --windowOperation, -wo {avg\|max\|sum} [optional]

    Window features can be computed by averaging, summing or taking the max across all bins within the window. Default: avg
   
* -iid, --internalInDir PATH

    path to directory where the internally used cooler matrices from createBaseFile are stored. Required when basefile was created 
    with -matrixFile option.
   
* --smooth FLOAT

    Standard deviation for gaussian smoothing of protein peaks; Zero means no smoothing. Default: 0.0

* --method {multiColumn, oneHot} [optional]

    Method to use for building the dataset.
    OneHot = 3 columns for start protein, window protein and end protein plus a one-hot vector encoding for the corresponding protein plus distance;
    MultiColumn = start, window and end for each protein separately, plus distance.
    Default: MultiColumn

* --removeEmpty BOOL [optional]

    Invalidate samples which have no protein data (no protein peaks / zero signal value for all genomic positions within a certain bin). Note that setting this option often leads to sparse datasets when using protein inputs in narrowPeak or broadPeak format. Default: False

* --noDiagonal, -nd INT [optional]

   Number of (side-)diagonals to ignore for training, Default: -1 (none), 0= main diagonal, 1= first side diagonal etc. Setting this option can remove outliers with extremly high interaction count values.
   Only useful when the basefile has been created with the -matrixFile option set.

* --printproteins, -pp BOOL [optional]

   Create plots for the resulting protein data (e.g. after scaling / thresholding). Output directory is the same as in -dod. Default: False.


* -bf, --baseFile PATH [required]

    Base file with binned proteins for given cell line and resolution, to be created with createBaseFile. 

* -chs, --chromosomes LIST OF TEXT [optional]
    
    If specifed, datasets are only computed for the chromosomes specified, otherwise for all numerical chromosomes. Note that all specified chromsomes
    must be present in the basefile, see above. Chromosomes must be specified
    by numbers only, e.g. '1' or '1,3,21'. Default: all numerical human
    chromosomes

* --help
    
    Show help message and exit.

Returns:
 * one .z-compressed dataset for each chromosome, stored in directory specified by -dod option.
 * Protein plots in the same directory as the datasets, if selected

### 3. training
This script builds and trains the regression model using a training dataset.
Training can only be performed if the underlying basefile has been created with a Hi-C matrix (-matrixFile option in createBasefile).


Options:
 * -tdf, --trainDatasetFile PATH [required]
 
    Hicprediction-dataset for training. Datasets are created by calling 'createTrainingSet' and have .z file extension

 * --noDist BOOL [optional]

    Do not use distance between start and end bin of each position as a feature to learn from. Default: False

 * --noMiddle BOOL [optional]

    Do not use window protein value between start and end bin of each position as a feature to learn from. Default: False

 * --noStartEnd BOOL [optional]

    Do not use signal values at the start and end bin of each position as a feature to learn from. Default: False

 * --weightBound1, -wb1 FLOAT [optional]

    Samples within [weightBound1...weightBound2] will be emphasized; only relevant if 'ovsF' > 0 and 'weightingType' != tads. Default 0.0

 * --weightBound2, -wb2 FLOAT [optional]
    
    Samples within [weightBound1...weightBound2] will be emphasized; only relevant if 'ovsF' > 0 and 'weightingType' != tads. Default 0.0

 * --ovsFactor, -ovsF FLOAT [optional]

    weight samples in range weightBound1...weightBound2 or samples within TADs such that 'ovsF'=(weight sum of weighted samples)/(weight sum of unweighted samples), 0.0 = no weighting. Default: 0.0

 * --tadDomainFile, -tads PATH [optional]

    TAD domain file in bed format for emphasizing samples within TADs; must be provided if 'weightingType' = tads

 * --weightingType, -wt {reads\|proteinFeatures\|tads} [optional]

    Compute weights for sample emphasizing based on reads, protein feature values or TADs, only relevant if 'ovsF' > 0. Default: reads

 * --featList, -fl LIST OF TEXT [optional]

    Name of features according to which the weights are computed; only relevant if 'weightingType' = proteinFeatures and 'ovsF' > 0.
    For example, when -fl 1,12,24 protein features generated from input files number 1, 12 and 24 will be considered for weight computation.

 * --plotTrees BOOL [optional]

    Plot decision trees.
    Setting this option increases runtime and is discouraged for standard settings of random forest. Default: False

 * --splitTrainset BOOL [optional]

    Split Trainingset to do a 5-fold Cross-Validation, i.e. return 5 trained models instead of 1. Default: False

 * --useExtraTrees BOOL [optional]

    Use sklearn.ensemble extra trees algorithm instead of random forests.
    Default: False
   
 * --modelOutputDirectory, -mod PATH [required]

    Output directory for model files

 * --conversion, -co {standardLog\|none} [optional]

    Conversion function for the interaction count values. Default: none

 * additional random forests or extra trees parameters

    most named parameters from sklearn.ensemble.RandomForestRegressor
    or sklearn.ensemble.ExtraTreesRegressor are supported as options, e.g.
    '-n_estimators 50' will set the number of trees to 50.
    See sklearn documentation for full parameter list: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

 * --help                        
 
    Show help message and exit.

Returns:
 * one or five .z-compressed trained model(s) with name dependent on chosen options, stored in directory given by -mod
 * Feature importance graph in the same directory as the model
 * Plots of decision trees in the same directory as the model, if selected


### 4. predict
This script predicts Hi-C matrices from the chosen model. It requires a test dataset (created by createTrainingSet) and a model (created by training) and will output a predicted matrix in cooler format.
Optionally a text file can be defined as output for evaluation metrics, if the test dataset contains target values.

Options:
 * -pod, --predictionOutputDirectory PATH [optional]
                                 
    Output directory for prediction files. If
     not set, no predicted Hi-C matrices are stored.
     Otherwise, the output will be a predicted Hi-C matrix in cooler format.

 * -rfp, --resultsFilePath PATH [optional]    
 
    File where evaluation metrics are stored. Must have a csv file extension (.csv). If not set, no evaluation is executed
 
 * -psp, --predictionSetPath PATH  [required]
    Test Data set used as input for prediction.
                                  
 * -mfp, --modelFilePath PATH [required]

    Model used for prediction  
 
 * -sigma FLOAT [0, 10] [optional]

    Smoothen the resulting matrix using gaussian filter with std. deviation sigma; default: 0.0, i.e., no smoothing

 * --noConvertBack FLAG [optional]

    Do not scale predictions according to input ranges. 
    Only relevant if -normalizeReadCounts True has been used with createTrainingSet. Default: not set

 * --help                          
 
    Show this message and exit.

Returns:
* Predicted matrix in cooler format in directory -pod, if specified
* csv-file, if specified (-rfp) and target values available 


### 5. Example Usage

Assume you have a Hi-C matrix for cell line GM12878, resolution 5kBp, and ChIP-seq data from RAD21, CTCF and H3K9me3 for both GM12878 and K562.
The directory structure might look as follows
```
./
./hg19.chrom.sizes
./Gm12878_Rad21.narrowPeak
./Gm12878_Ctcf.broadPeak
./Gm12878_H3k9me3.bigwig
./K562_Rad21.narrowPeak
./K562_Ctcf.broadPeak
./K562_H3k9me3.bigwig
./hicMatrices/
  ./hicMatrices/Gm12878_5kb.cool
```

You could then use hicprediction to train a random forest on the GM12878 matrix and GM12878-ChIP-seq data and then use this model to predict the Hi-C matrix of K562 using K562-ChIP-seq data.
```
#create a basefile for GM12878, to be used for training
#Hi-C matrix is specified as well as storage path iod
#resolution 5000bp, proteins RAD21, CTCF and histone H3k9me3
#the matrices for all numerical chromosomes will be stored in the given iod directory
$ createBaseFile -bf Gm12878_5kb.ph5 -ct Gm12878 -r 5000 -csf ./hg19.chrom.sizes -mf ./hicMatrices/Gm12878_5kb.cool -iod ./hicMatrices/ Gm12878_Rad21.narrowPeak Gm12878_Ctcf.broadPeak Gm12878_H3k9me3.bigwig 

#create a basefile for K562, to be used as prediction target
#no Hi-C matrix and no -iod required
#resolution is the same as in the training case. 
#proteins, too, are the same, and in the same order as in the training case above, just for K562 instead of GM12878. It is recommended to also use the same input formats. 
#chromosomes must be the ones we want predicted later on
#here only chr2 for simplicity (specified w/o chr prefix)
$ createBaseFile -bf K562_5kb.ph5 -ct K562 -r 5000 -chs 2 -csf ./hg19.chrom.sizes K562_Rad21.narrowPeak K562_Ctcf.broadPeak K562_H3k9me3.bigwig

#create dataset for training
#use the basefile created above and the output directory of
#the Hi-C matrices as inputs to -bf and -iid, respectively.
#window size set to 500 just as an example
$createTrainingSet -bf Gm12878_5kb.ph5 -dod ./ -iid ./hicMatrices/ -ws 500

#create dataset for prediction (test set) 
#window size must be the same as for the training dataset
#-iid is not required here, since no matrix was passed when
#creating the basefile
$ createTrainingSet -bf K562_5kb.ph5 -dod ./ -ws 500

#build and train the model
#createTrainingSet outputs a dataset for all chromosomes
#since the test set has been created for chr2 only, we also
#use the train data set for chr2
#using default params as a start
$ training -tdf GM12878_20000_Mavg_Wavg500_Bchr2.z -mod ./ 

#predict target matrix from model using input from test set
#take the model created by 'training' for GM12878 and the dataset 
#created by 'createTrainingSet' for K562
#note that -rfp cannot be used, since the target values for K562 have
#not been provided in 'createBaseFile'.
$ predict -psp K562_20000_Mavg_Wavg500_Bchr2.z -mfp GM12878_20000_Mavg_Wavg500_Bchr2_Cnone.z -pod ./

#after this completes, you should have the predicted matrix
#Model_GM12878_20000_Mavg_Wavg500_Cnone_Bchr17_PredictionOn_K562_chr17.cool 
#in your -pod directory ./

```

You might then want to use e.g. pygenometracks to plot the predicted matrix vs. the real matrix (if you have one) and/or vs. the protein inputs.