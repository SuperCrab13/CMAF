# CMAF: An Interpretable Cross-Modal Adversarial Fusion Method for Cancer Analysis
CMAF is an multi-modal fusion method utilized to fuse pathology data and multi-omic data

## Data Preparing
WSI data and omic data can be downloaded from the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/) and [UCSC Xena](https://xenabrowser.net/).

## Running 
Running the following command to train CMAF and perform survival predict
``` 
python main.py --split_dir tcga_CANCER_TYPE --fusion bilinear --data_root_dir PATH_TO_WSI --result_dir PATH_OF_RESULT --apply_mutsig --reg_type omic --lr 4e-4 --alignment --model_type cmaf
``` 
5-fold split files are prepared in ./splits/5foldcv <br />
