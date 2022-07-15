# NormVAE Paper Implementation
Paper link: [Link to the NormVAE paper](https://arxiv.org/pdf/2110.04903.pdf)

Implementation of NormVAE Paper using PyTorch Framework on Custom Dataset(ADNI Dataset).

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

* To generate the reconstructed samples along with training the model:
  * number of samples can be given using --gensamples
```
python reconstruct.py --gensamples #no.ofsamples --output_format {outputformat as string}
```

### An example usage (with default values):
#### Only For Training the Model
```
python train_model.py --epochs 1500 --bsize 1024
```
#### For Training the Model and Generating reconstructed samples
* To save the reconstructed samples in excel file:  
```
python reconstruct.py --gensamples 20 --output_format "xlsx"
```
* To save the reconstructed samples as csv file:
```
python reconstruct.py --gensamples 20 --output_format "csv"
```

**Note:** The above command performs training of the model and generates reconstructed samples from the decoder part of the VAE in an excel file with name `reconstruct.xlsx` or `reconstruct.csv` if you choose the output_format as `csv`. Default output format is `xlsx`.  

* For more help on how to use the model with different hyperparameters: 

```
python train_model.py -h
```



### References: 

```
sayantan.k (2022) NormVAE: Normative modelling on neuroimaging data using Variational Autoencoders, arXiv:2110.04903v2 [eess.IV] 30 Jan 2022
```
