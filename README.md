# Rossmann Sales Prediction
This repository describes the Mini Competition run in Data Science Retreat - Batch 25 between 2-4 February 2021, by the Team 3 composed by:
* Gert-Jan Dobbelaere
* Sergio Vechi
* Alberto Julián

## Introduction
This DSR mini-competition is based on a Kaggle competition which run from Sep 30, 2015 to Dec 15, 2015:
https://www.kaggle.com/c/rossmann-store-sales/overview/timeline

### Generic objectives of the competition
* Develop an end-to-end Data Science project
* Live the experience of working as a Data Science team, sharing python code and jupyter notebooks through Github
* Present results to an audience

### Specific objectives of the competition
The Rossmann competition aims to predict sales on more than a thousand stores based on historic sales and additional information provided.

### Scoring Criteria

The competition is scored based on a composite of predictive accuracy, following a metric detailed below, and reproducibility.

### Information provided
Two csv files were provided for training the models:
* train.csv
* store.csv

Both datasets are described below.

Additionally, a test dataset was provided to check the accuracy of the models. The holdout test period is from 2014-08-01 to 2015-07-31 - the holdout test dataset is the same format as `train.csv`, and is called `holdout.csv`.

### 

## Content of the repository
### Jupyter notebooks

### Python files

## Installation instructions

##* Predictive accuracy

The task is to predict the `Sales` of a given store on a given day.

Submissions are evaluated on the root mean square percentage error (RMSPE):

![](./assets/rmspe.png)

```python
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
```

More info from Kaggle:

```
Id - an Id that represents a (Store, Date) duple within the test set

Store - a unique Id for each store

Sales - the turnover for any given day (this is what you are predicting)

Customers - the number of customers on a given day

Open - an indicator for whether the store was open: 0 = closed, 1 = open

StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None

SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

StoreType - differentiates between 4 different store models: a, b, c, d

Assortment - describes an assortment level: a = basic, b = extra, c = extended

CompetitionDistance - distance in meters to the nearest competitor store

CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened

Promo - indicates whether a store is running a promo on that day

Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
```

The holdout test period is from 2014-08-01 to 2015-07-31 - the holdout test dataset is the same format as `train.csv`, as is called `holdout.csv`.

After running `python data.py -- test 1`, the folder `data` will look like:

```bash

