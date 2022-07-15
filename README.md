# NormVAE Paper Implementation
Paper link: [Link to the NormVAE paper](https://arxiv.org/pdf/2110.04903.pdf)

Implementation of NormVAE Paper using PyTorch Framework on Custom Dataset.

## Usage:

### Installing Dependencies:
```
pip install -r requirements.txt
```

The following packages will be installed : 
* PyTorch
* Numpy
* Pandas
* Scikit-Learn

### Dataset: 
[Dataset in Excel Format](https://github.com/sandeshkatakam/NormVAE-Neuroimaging/blob/main/ADNI_sheet_for_VED.xlsx)

### Training the Model:  
* Download the code from the repository and get the dataset file or modify the dimensions in the code for your dataset.  
* After downloading the repository from the terminal go to the directory and Install depedencies using : 
```
pip install -r requirements.txt
```
* Train the model using the dataset specifying the Number of Epochs and the Batch size like described below:

```
python train_model.py --epochs #epochs --bsize #batchsize
```
* For more help on how to use the model with different hyperparameters: 

```
python train_model.py -h
```

### Generating Reconstructions from the Model:
Use the Notebook to generate the reconstructions (TO be updated soon in the repository)

### References: 

```
sayantan.k (2022) NormVAE: Normative modelling on neuroimaging data using Variational Autoencoders, arXiv:2110.04903v2 [eess.IV] 30 Jan 2022
```
