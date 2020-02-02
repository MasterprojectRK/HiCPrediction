# HiCPrediction
HiCPrediction allows predicting Hi-C matrices using protein levels given by ChIPseq data. 
It is based on random forest regression as proposed by [Zhang, Chasman, Knaack and Roy in 2018](http://dx.doi.org/10.1101/406322). 

HiCPrediction consists of four steps - binning the proteins, creating training/test data sets, training the regression model and prediction, which are described below in more detail. 


## Installation
HiCPrediction requires Python3.6, hicmatrix, hicexplorer, hic2cool, pybedtools, pybigwig, pydot and graphviz.
It is recommended to install everything via conda into an empty environment:

```
$ conda install -c conda-forge -c bioconda -c ska hicmatrix hicexplorer
$ conda install -c conda-forge -c bioconda -c ska hic2cool
$ conda install -c conda-forge -c bioconda -c ska pybedtools pybigwig pydot
$ conda install -c abajorat hicprediction
```
## Usage

### 1. createBaseFile.py
This script bins the proteins for the given cell line and resolution.
It creates a base file and also per-chromosome cooler matrices, if provided with a HiC-Matrix. 
Basefiles are required for building training and test datasets (see below).
Basefiles have to be created for each cell line and each resolution, using the appropriate protein files.

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
depending on the resolution of the HiC-matrix. Must be used if matrixfile is specified


 * --help                   
 
    Show this message and exit.
  
* Positional arguments: 

    List of protein peaks in narrowPeak, broadPeak or bigwig format. The proteins must be the same for training / test datasets and they must be in the same order, see examples.

### 2. createTrainingSet.py
This script creates the datasets that are required later both for training the regression model and for predicting HiC-matrices from given models. 

* -bf, --baseFile PATH [required]

    Base file with binned proteins for given cell line and resolution, can be created with createBaseFile. 
* -iid, --internalInDir PATH [required]

    path to directory where the internal matrices from createBaseFile are stored
* -ws, --windowSize INT [required]
    
    Maximum bin distance which will be considered for learning and prediction [default: 200; corresponds to 1.000.000 bp at 5kb resolution]
* -wo, --windowOperation {avg\|max\|sum} [required]

    How the proteins between two bins should be considered during binning [default: avg]

* --ignoreCentromeres BOOL [required]
    
    Cut out the centroid arms for training [default: True]
* --normalize BOOL
    
    normalize protein signal values to a 0-1 range [default: False]
* -mo, --mergeOperation {avg\|max} [required]
    
    This parameter defines how the proteins are binned [default: avg]
* -chs, --chromosomes LIST OF TEXT [optional]
    
    If specifed, datasets are only computed for the chromosomes specified, otherwise for all 
    chromosomes in the given basefile
 
* -dod, --datasetOutputDirectory PATH [required]

    Output directory for training set files
* -cmf, --centromeresFile PATH [required]

    Text file containing centromer regions. See centromeres.txt in InternalStorage for formatting
* --cutoutLength 

    currently undocumented
* --smooth FLOAT [optional]

    std. deviation for gaussian smoothing of protein inputs
    default=0.0, which means no smoothing

* --method {multiColumn, oneHot} [required]

    Method to use for building the dataset.
    OneHot = 3 columns for start protein, window protein and end protein plus a one-hot vector encoding for the corresponding protein plus distance;
    MultiColumn = start, window and end for each protein separately, plus distance.
    Default = MultiColumn
* --help
    
    Show this message and exit.

### 3. training.py
This script builds and trains the regression model. A training set as created above by createTrainingSet.py must be given as well as the output directory for the model.
Optionally, the ChIP-seq read values can be log-converted.
Training can only be performed if the underlying basefile has been created with a HiC-matrix


Options:
 * -tdf, --trainDatasetFile PATH [required]
 
    File from which training data is loaded
                                  
 * -co, --conversion {standardLog|none} [required]
    
    Define a conversion function for the read
    values  [default: none]

 * -mod, --modelOutputDirectory PATH [required]
     
    Output directory for model files  
 * --trees
 * --maxfeat
 * --nodist
 * --nomiddle
 * --nostartend
 * --ovspercentage
 * --ovsfactor
 * --ovsbalance
 * --help                        
 
    Show this message and exit.

### 4. predict.py
This script predicts HiC-matrices based on the chosen model. It requires a test dataset (created above by createTrainingSet.py) and a model (created above by training.py) and will output the predicted matrix in cooler format.
Optionally a CSV-file can be defined as output for evaluation metrics, if the test dataset contains target values.

Options:
 * -rfp, --resultsFilePath PATH     
 
    File where evaluation metrics are stored. Must have a csv file extension (.csv). If not set, no evaluation is executed
 * -pod, --predictionOutputDirectory PATH
                                 
    Output directory for prediction files. If
     not set, no predicted Hi-C matrices are stored.
     Otherwise, the output will be a predicted HiC-matrix in cooler format.
 * -psp, --predictionSetPath PATH  [required]
    Test Data set used as input for prediction.
                                  
 * -mfp, --modelFilePath PATH [required]

    Model used for prediction  

 * --help                          
 
    Show this message and exit.

### 5. Example Usage

Assume you have a HiC matrix for cell line GM12878, resolution 5kBp, and ChIP-seq data from RAD21, CTCF and H3K9me3 for both GM12878 and K562.

You could then use HiC-Prediction to predict the HiC-matrix 
of K562 from the given GM12878 HiC matrix.

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

#create datasetfor prediction (test set) 

#build and train the model

#predict target matrix from model using input from test set

```
