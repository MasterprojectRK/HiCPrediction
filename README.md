# HiCPrediction
HiCPrediction enables the training of Random Forest Regressor models for Hi-C matrices by using protein levels. The models can then be used to predict Hi-C matrices of other chromosomes and cell lines. The current version requires an existing Hi-C matrix, or at least the binning intervals for any prediction. The framework has 2 data preparation scripts, one training scrip and one prediction script. All of them have to be executed consecutively in order to work. 
## Installment
Unordered lists can be started using the toolbar or by typing `* `, `- `, or `+ `. Ordered lists can be started by typing `1. `.

## Scripts

### 1. createBaseFile'
This script creates bins the proteins for the given cell line and creates a base file that is needed for the consecutive steps. This step has to be executed for each cell line or resolution with the according protein files.
The following arguments can be passed
* -bf, --basefile(required) --  output path for base file  (ph5)
* -mf,  --matrixfile(required) -- path to input HiC matrix (cool)                                                        
* -ct, --celltype(required) -- cell line of the input matrix                                                
* -r, --resolution(required) -- resolution of the input matrix                                          
* -chs, --chromosomes(optional) -- comma separated list of chromosomes to be processed, if not set all chromosomes will be choosen
* proteinfiles(required) -- list of paths of the protein files to be processed  (narrowpeak)      


Example:
$ createBaseFile -mf hic.cool -bf basefile.ph5 -ct Gm12878 -r 5000 Gm12878_Rad21.narrowpeak Gm12878_Ctcf.narrowpeak
