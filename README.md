# AutoRec: Autoencoders Meet Collaborative Filtering

## Introduction

AutoRec is a type of Collaborative Filtering model based on autoencoder paradigm. AutoRec try to predict rating for This respository implement AutoRec model in Pytorch.

## How to use this respository

1. Clone this project to current directory. Using those commands:
```
!git init
!git remote add origin https://github.com/tuanio/autorec
!git pull origin main
```
2. Install requirement packages
```
!pip install -r requirements.txt
```

3. Edit `configs.yaml` file for appropriation.
4. Train model using `python main.py -cp conf -cn configs`

## Note

- Kaggle kernel that running this respository: https://www.kaggle.com/code/tuannguyenvananh/autorec-for-movielens-1m

## Environment variable
- `HYDRA_FULL_ERROR=1`