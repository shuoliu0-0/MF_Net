# MF_Net

## Introduction
* Source code for the paper "A Unified Multimodal Fusion Framework for Drug-Target Affinity Prediction: From Benchmark Performance to Nanomolar Lead Discovery".

* We propose a Multimodal Fusion Network (MF-Net) that hierarchically incorporates sequence-, atom-, and fragment-level representations to capture global, local, and functional interactions across multiple scales.

![Multimodal Fusion Network](images/fig1.png)


## Dataset
The PDBbind v2016 and CASF-2016 datasets are downloaded from https://www.pdbbind-plus.org.cn/download. The complete DUD-E benchmarking set is available at http://dude.docking.org. And LIT-PCBA is available at http://drugdesign.unistra.fr/LIT-PCBA. The constructed 3D kinase-drug binding affinity datasets, 3DKKIBA, is available at https://github.com/Yanara-Tian/MMCLKin/tree/main/datasets/3DKKIBA.

## Environment
* Base dependencies:
```
  - numpy == 1.21.5
  - rdkit == 2018.03.4
  - pandas == 1.3.5
  - python == 3.7.16
  - pytorch == 1.12.1
  - scikit-learn == 1.0.2
```

## Usage

### Process data
* Extracting the features of PDBBind v2016 and CASF-2016 datasets.
```bash
python data/process.py
```

### train
#### Args:
- --config : The path of config file.
- --data_dir : The path of input file.
- --save_dir : The path to save output model.
- --batch_size : The input batch size for training.
- --epochs : The number of epochs to train.
- --early_stop_epoch : The number of early stop epoch to train.
- --lr : The learning rate for the prediction layer.

#### Quick Run
```bash
python run_train.py
```

### Reproduce Results
#### Binding affinity prediction
* Predictive performance of binding affinity on the PDBBind v2016 dataset.
```bash
python run_test.py
```
```
RMSE     MAE     Rp      SD
1.125   0.859   0.876   1.050
```