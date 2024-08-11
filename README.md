# CMAF: An Interpretable Cross-Modal Adversarial Fusion Method for Cancer Analysis
CMAF is an multi-modal fusion method utilized to fuse pathology data and multi-omic data

## Data Preparing
WSI data and omic data can be downloaded from the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/) and [UCSC Xena](https://xenabrowser.net/). We used the publicaly available [CLAM WSI-analysis toolbox](https://github.com/mahmoodlab/CLAM) to process WSI data (.svs format). 256x256 patches are first extracted from tissue regions of each WSI, and a pretrained res-Net50 is utilized to encode image patches into 1024-dim feature vector. The extracted features are saved as torch tensors of size Nx1024, where N is the number of patches for each WSI. These feature matrices are saved in .pt files and saved as following structure.

```bash
DATA_DIRECTORY
    └──tcga_brca_20x_features/
          └──pt_files/
            └──slide1.pt
            └──slide2.pt
            └──.......
    └──tcga_gbmlgg_20x_features/
          └──pt_files/
            └──slide1.pt
            └──slide2.pt
            └──.......
    └──tcga_kirc_20x_features/
          └──pt_files/
            └──slide1.pt
            └──slide2.pt
            └──.......
```

## Running 
Running the following command to train CMAF and perform survival predict
``` 
python main.py --split_dir tcga_CANCER_TYPE --fusion bilinear --data_root_dir PATH_TO_WSI --result_dir PATH_OF_RESULT --apply_mutsig --reg_type omic --lr 4e-4 --alignment --model_type cmaf
``` 
5-fold split files are prepared in ./splits/5foldcv <br />
