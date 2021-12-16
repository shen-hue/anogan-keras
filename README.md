# SHAP of AnoGAN

Use SHAP value to explain AnoGAN model


## Usage  

1. activate virtual environment
```
. /home/lin/src/.pyvenv/anogan-keras/bin/activate
```
2. train and test the AnoGAN model
```
python main.py
# result will automaticly save in folder result_artificial
```
3. calculate SHAP value
```
python SHAP.py
# result will automaticly save in folder result_artificial
```
4. calculate F1 score and confusion matrix of AnoGAN model
```
python precision.py
```
