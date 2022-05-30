# Hurricane Path Prediction : Next Frame Generation Using Machine Learning

![image](https://user-images.githubusercontent.com/88569855/170634319-3c4ab40f-6917-45c3-8f73-90c00cfcc681.png)

-------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Objective 
Given satellite images of an active hurricane, this software tool is capable of predicting `n` number of future images using a spatio-temoral method - the convoltuional LSTM. This tool , written in python using the pytorch framework is capable of predicting images with a SSIM Index( an index that measures similarity with the original path), consistently ***over 50%*** for up to 10 future image predictions. 

The main file `Team Ivan : Hurricane Prediction using Conv LSTM` contains the final model based on the 'nasa_tropical_storm_competition' dataset avaible at Radiant MLHub — Open Geospatial ML Library. This is a free dataset availabe for downlad upon registration. The notebooks use APi access keys linked to the contributor's account, however you may generate your own API authentication using this [link](https://radiantearth.auth0.com/login?state=hKFo2SB2R3lweUF6V0I0bUdrTEwxRXlyVHRHUmhQUkNsTnJIeaFupWxvZ2luo3RpZNkgMDctd1ozM1gzNHpEdXVVaDBXYlh4eHhycTZiZGk1Q1qjY2lk2SBQM0lxTHFiWFJtMTBCVUpNSFhCVXRlNlNBRG4wUzhEZQ&client=P3IqLqbXRm10BUJMHXBUte6SADn0S8De&protocol=oauth2&scope=openid%20profile%20email&response_type=code&redirect_uri=https%3A%2F%2Fmlhub.earth%2Fapi%2Fauth%2Fcallback&audience=https%3A%2F%2Fapi.radiant.earth%2Fv1&nonce=tYY1JM5kz6Qx7sOe_5CHpL5TpLdze3CKRQyEcfDXNzc&code_challenge=Tn_-Tp1Qqpj6jcWFtvlPed5zftmGLSo7h83S2SvN4gI&code_challenge_method=S256)


-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Installation Guide 

Create the environment using
```bash
  conda env create -f environment.yml
```
Activate the environment 
```bash
  conda activate hurricane
```

-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Models and Metrics in the Tool

**Basic** : Recurrent Neural Network Model, Long Short Term Memory Network Model \
**Advanced** : Convolutional LSTM

The metrics for tool accuray and quality of output predictions include Mean Squared Erro(MSE) and the  Structural Similarity Index(SSIM). See the results below for an understanding of the tool's output accuracies.

-------------------------------------------------------------------------------------------------------------------------------------------------------------
#  Folder Structure

Notebooks Available :
- `Team Ivan : Hurricane Prediction using Conv LSTM.ipynb` :  contains our best model and the results on the surprise storm
- `Team Ivan : Basic Models.ipynb` : Second notebook contains the different strategies we implemented and some observations and comments
- `preprocessing.ipynb` : The preprocessing file used if the data set is not sufficiently large

------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instructions for Use

Note: the parameters you need to update in the notebooks will be marked with an !! UPDATE HERE !! You can use this as a reference to update the necessary values

For each of the ipynb files, run the cells in sequence and make updates as required in the cells that will allow redefinition of parameters based on your data
To test this model on your own Storm Dataset :

## **Step 1 :** Import the required packages. 
Run cells to import the required packages and connect to your Google Drive Account to ensure you can save your model.


## **Step 2 :** Loading the Dataset

1 . For Demo Data in the ML Radient Hub update your API key :
  
```
os.environ['MLHUB_API_KEY'] = 'INSERT YOUR API KEY HERE'
```
   Update the download drive to your download path 
```
download_dir = Path('ADD YOUR PATH').resolve()
```
   Update the following lines with your own path to the dataset to ensure correct extraction of data:
    
```
train_source = 'NAME OF TRAIN SOURCE FILE'
train_labels = 'NAME OF LABELS FILE'
test_source = 'NAME OF TEST LABELS FILE'
```
2 . For a new storm :
```
data_path = 'PATH FOR YOUR DATA'
```

## **Step 3 :** Select the Hyper Parameters for your model : 
As mentioned the notebook has markings and directions along with an indexing for each of these values. The selections include : \
    - Resize of Images : `resize_value`\
    - Number of Frames : `frames`\
    - Preprocessing Style : `preprocessing.ipynb` \
    - Number of Images to Predict : `images_to_predict` \
If you wish, after your run, to further customize your model you can upate the hyperparameters for the model in the Hyperparamaters code block linked in the index of the notebooks

## **Step 4 :** Get the outputs and metrics for the image predictions
If you are making further predictions. please select the correct array values to plot and measure the image metric. The inputs are triple indexed and the outputs are double indexed. Once they are defined accurately, the plots and the metrics will be calculated automatically.


------------------------------------------------------------------------------------------------------------------------------------------------------------
# Example Results

<img width="660" alt="image" src="https://user-images.githubusercontent.com/88569855/170668579-a8c084c1-9380-43fa-92f6-d401efa12016.png">

Metrics 

<img width="462" alt="image" src="https://user-images.githubusercontent.com/88569855/170660863-d188e9a3-59dc-4a7f-88e4-d1d6f3c521ef.png">

------------------------------------------------------------------------------------------------------------------------------------------------------------
# Documentation

The automatic documentation generated in the html formal can be referred to in the`docs` folder [here](#docs)

------------------------------------------------------------------------------------------------------------------------------------------------------------
# License

The License and the permissions can be viewed [here](#License)

