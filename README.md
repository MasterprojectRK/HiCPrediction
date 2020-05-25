# HiCPrediction
HiCPrediction allows predicting Hi-C matrices using protein levels given by ChIPseq data. 
It is based on random forest regression as proposed by [Zhang, Chasman, Knaack and Roy in 2018](http://dx.doi.org/10.1101/406322). 

HiCPrediction consists of four steps
* binning the proteins, 
* creating training and test data sets, 
* training a regression model on given HiC-matrices and ChIP-seq data,
* prediction of unknown HiC-matrices from ChIP-seq data

These four steps are described below in more detail. 


## Installation
HiCPrediction requires Python >= 3.6, click, cooler, graphviz,
h5py, hicmatrix, hyperopt, joblib, matplotlib, numpy, pandas,
pybedtools, pyBigWig, pydot, scikit-learn, scipy and tqdm.

It is recommended to install hicprediction using conda and pip
into a fresh conda environment:

```
$ conda create -n hicprediction
$ conda activate hicprediction
$ conda install hicexplorer
$ pip install https://github.com/MasterprojectRK/HiCPrediction/hicprediction.whl

```
## Usage

### 1. createBaseFile
This script bins the proteins for the given cell line and resolution.
It creates a base file and also per-chromosome cooler matrices, if provided with a HiC-Matrix. 
Basefiles are required both for building training and test datasets (see below. Basefiles can span several or all numerical chromosomes, but separate files have to be created for each cell line and each resolution, using the appropriate protein files.

Options:
 * -bf, --baseFile PATH [required]
    
    Name of Base file to be created, should end in .ph5 
 
 * -ct, --cellType TEXT [required]      
 
    User-defined cell type for analysis and documentation
                            
 * -r, --resolution INT [required]    
 
    Resolution in basepairs, must match bin size of HiC-matrix (if provided)
                            
 * -chs, --chromosomes LIST OF TEXT [optional]
 
    Optional; if set, binning is done only for the
chromosomes specified, otherwise for all numerical ones. Must be a comma separated list without chr, e.g. 1,3,21
 * -csf PATH [required]  

    chromosome size file, see e.g. hg19.chrom.sizes for formatting. Required for binning.

 * -mf, --matrixFile PATH [optional]  
 
    HiC-matrix in cooler format, required if basefile is to be used for training later in the process (see below) 

 * -iod, --internalOutDir PATH [optional]  
 
    Path where internally used matrices will be stored. 
Note that these matrices can be several 100's of MB in size
depending on the resolution of the HiC-matrix. Must be used if matrixFile is specified


 * --help 

   show help and exit
  
* Positional arguments: 

    List of protein peaks in narrowPeak, broadPeak or bigwig format. The proteins must be the same for training / test datasets and they must be in the same order, see examples.

Returns:
 * basefile with user-defined name (see option -bf)

### 2. createTrainingSet.py
This script creates the datasets that are required later both for training the regression model and for predicting HiC-matrices from given models. 

* -bf, --baseFile PATH [required]

    Base file with binned proteins for given cell line and resolution, to be created with createBaseFile. 
* -iid, --internalInDir PATH [required]

    path to directory where the internal matrices from createBaseFile are stored
* -ws, --windowSize INT [optional]
    
    Maximum bin distance which will be considered for learning and prediction [default: 200; corresponds to 1.000.000 bp at 5kb resolution]
* -wo, --windowOperation {avg\|max\|sum} [optional]

    Window features can be computed by averaging, summing or taking the max across all bins within the window [default: avg]


* -chs, --chromosomes LIST OF TEXT [optional]
    
    If specifed, datasets are only computed for the chromosomes specified, otherwise for all 
    chromosomes in the given basefile
 
* -dod, --datasetOutputDirectory PATH [required]

    Output directory for training set files

* --smooth FLOAT [0, 10] [optional]

    std. deviation for gaussian smoothing of protein inputs
    default=0.0, which means no smoothing

* --method {multiColumn, oneHot} [optional]

    Method to use for building the dataset.
    OneHot = 3 columns for start protein, window protein and end protein plus a one-hot vector encoding for the corresponding protein plus distance;
    MultiColumn = start, window and end for each protein separately, plus distance.
    Default: MultiColumn
* --help
    
    Show this message and exit.

Returns:
 * .z-compressed dataset for each chromosome in directory -dod, with name dependending on the chosen options

### 3. training.py
This script builds and trains the regression model. A training set as created above by createTrainingSet.py must be given as well as the output directory for the model.
Optionally, the ChIP-seq read values can be log-converted.
Training can only be performed if the underlying basefile has been created with a HiC-matrix


Options:
 * -tdf, --trainDatasetFile PATH [required]
 
    File from which training data is loaded
                                  
 * -co, --conversion {standardLog|none} [optional]
    
    conversion function for the read
    values; default: none

 * -mod, --modelOutputDirectory PATH [required]
     
    Output directory for model files  
 * --trees UINT [optional]

    number of trees in the random forest; default: 10
 * --maxfeat {sqrt|none} [optional]

    The number of features to consider when looking for the best split in the random forest (see sklearn). Default: none, i.e. use all features, if required
 * --nodist BOOL [optional]

    Do not use distance between start and end bin of each position as a feature to learn from. Default: False
 * --nomiddle BOOL [optional]

    Do not use window protein value between start and end bin of each position as a feature to learn from. Default: False
 * --nostartend BOOL [optional]

    Do not use signal values at the start and end bin of each position as a feature to learn from. Default: False
 * --ovspercentage FLOAT [0, 1] [optional]

    Split the samples at ovspercentage times max. read count in train matrix and oversample the upper part.
    Default: 0.0, i.e., no oversampling

 * --ovsfactor FLOAT [0.0, inf) [optional]

    add the samples selected by ovspercentage to the original dataframe such that (number of samples in upper range) / (number of samples in lower range) >= ovsfactor. Default: 0.0, i.e., no oversampling

 * --ovsbalance BOOL [optional]

    Balance the samples in the oversampled range, such that they are equally frequent. Only effective, if ovsfactor > 0.0 and ovspercentage in (0,1). Default: False, i.e., no balancing

 * --help                        
 
    Show this message and exit.

Returns:
 * .z-compressed trained model with name dependent on chosen options, stored in directory given by -mod
 * Feature importance graph (.png) in the same directory as the model

### 4. predict.py
This script predicts HiC-matrices from the chosen model. It requires a test dataset (created by createTrainingSet.py, see above) and a model (created by training.py, see above) and will output the predicted matrix in cooler format.
Optionally a CSV-file can be defined as output for evaluation metrics, if the test dataset contains target values.

Options:
 * -pod, --predictionOutputDirectory PATH [optional]
                                 
    Output directory for prediction files. If
     not set, no predicted Hi-C matrices are stored.
     Otherwise, the output will be a predicted HiC-matrix in cooler format.

 * -rfp, --resultsFilePath PATH [optional]    
 
    File where evaluation metrics are stored. Must have a csv file extension (.csv). If not set, no evaluation is executed
 
 * -psp, --predictionSetPath PATH  [required]
    Test Data set used as input for prediction.
                                  
 * -mfp, --modelFilePath PATH [required]

    Model used for prediction  
 
 * -sigma FLOAT [0, 10] [optional]

    Smoothen the resulting matrix using gaussian filter with std. deviation sigma; default: 0.0, i.e., no smoothing

 * --help                          
 
    Show this message and exit.

Returns:
* Predicted matrix in cooler format in directory -pod, if specified
* csv-file, if specified (-rfp) and target values available 


### 5. Example Usage

Assume you have a HiC matrix for cell line GM12878, resolution 5kBp, and ChIP-seq data from RAD21, CTCF and H3K9me3 for both GM12878 and K562.

You could then use HiC-Prediction to train a random forest on the GM12878 matrix and GM12878-ChIP-seq data and then use this model to predict the HiC matrix of 
of K562 from K562-ChIP-seq data.

```
#create a basefile for GM12878, to be used for training
#HiC-matrix is specified as well as storage path iod
#resolution 5000bp, proteins RAD21, CTCF and histone H3k9me3
#the matrices for all numerical chromosomes will be stored in the given iod directory
$ createBaseFile.py -bf Gm12878_5kb.ph5 -ct Gm12878 -r 5000 -csf ./hg19.chrom.sizes -mf Gm12878_5kb.cool -iod ./hicMatrices/ Gm12878_Rad21.narrowPeak Gm12878_Ctcf.broadPeak Gm12878_H3k9me3.bigwig 

#create a basefile for K562, to be used as prediction target
#no HiC-matrix and no iod required
#resolution is the same as in the training case. 
#proteins, too, are the same, and in the same order as in the training case above, just for K562 instead of GM12878. It is recommended to also use the same input formats. 
# chromosomes must be the ones we want predicted later on
# here only chr2 for simplicity (specified w/o chr prefix)
$ createBaseFile.py -bf K562_5kb.ph5 -ct K562 -r 5000 -chs 2 -csf ./hg19.chrom.sizes K562_Rad21.narrowPeak K562_Ctcf.broadPeak K562_H3k9me3.bigwig

#create dataset for training
#use the basefile created above and the output directory of
#of the HiC-matrices as inputs to -bf and --iid, respectively.
#window size set to 500 just as an example
$createTrainingSet.py -bf Gm12878_5kb.ph5 -dod ./ -iid hicMatrices/ -ws 500 --ignoreCentromeres False

#create dataset for prediction (test set) 
#window size must be the same as for the training dataset
#iid is not required here, since no matrix was passed when
#creating the basefile
$createTrainingSet.py -bf K562_5kb.ph5 -dod ./ -ws 500 --ignoreCentromeres False

#build and train the model
#createTrainingSet.py outputs a dataset for all chromosomes
#since the test set has been created for chr2 only, we also
#use the train data set for chr2
#using default params as a start is usually a good idea 
$training.py -tdf GM12878_20000_Mavg_Wavg500_Bchr2.z -mod ./ 

#predict target matrix from model using input from test set
#use the model output from training.py for GM12878 and the dataset created by createTrainingSet.py for K562
$predict.py -psp K562_20000_Mavg_Wavg500_Bchr2.z -mfp GM12878_20000_Mavg_Wavg500_Bchr2_Cnone.z -pod ./

#after this completes, you should have the predicted matrix
#Model_GM12878_20000_Mavg_Wavg500_Cnone_Bchr17_PredictionOn_K562_chr17.cool 
#in your -pod directory ./

```

You might then want to use e.g. pygenometracks to plot the predicted matrix vs. the real matrix (if you have one) and/or vs. the protein inputs.