# About this repository
In this repository we try to show simple example related to machine learning but also data preprocessing.  
Some exercises may include the following:
- How to deal with imbalanced datasets?
- How to deal with missing values?
- How to deal and detect outliers?
- How to reduce the feature space?
- How to deal with categorical variables?


## Classification
File: *KNN_heart_attack.ipynb*

The first exercise is done on medical data in order to predict if a patient's heart is potentialy at risk or not using a supervised learning clustering algorithm called K-NN.  

In statistics, the k-nearest neighbors algorithm (k-NN) is a non-parametric classification method.  
It is used for classification and regression.  
  
In k-NN classification, the output is a class membership. An object is classified by a plurality vote
of its neighbors, with the object being assigned to the class most common among its k nearest neighbors
(k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of
that single nearest neighbor (source: [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)).  

The dataset can be downloaded on Kaggle [here](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) and the related jupyter notebook is the file called *heart_attack_knn.ipynb*.

### Workflow
1. Load the dataset
2. Clean the dataset
3. Exploratory Data Analysis (EDA)
4. Feature selection
5. Normalize/Standardized continous variable
6. Model
7. Grid Search cross-validation for hyperparmeter tuning
8. Predict out-of-sample


## Feature space reduction
File: *AE_feature_space_reduction.ipynb*

Some benefits, among others, of dimension reduction include:
- Less storage needed
- Less training time
- Mitigate multicolinearity
- Least but not last may also improve the subsequent alogrithms performance

We know that we can use, for example, a PCA to reduce the feature space but we can also use a Neural-Network algorithm, in particular the **AutoEncoder** (**AE**).
Indeed after training, we can use the *encoder* part of the **AE** to reduce our input dimension. In the first part of the notebook we reduce the input dimension from **30 to 8 features** then in the second part we also show how to use the latent space aka compressed information as *new* input for a classifier.

In order to showcase the concept, we use a highly imbalanced synthetic dataset and a gradient boosting decision tree.  
The main libraries are `scikit-learn` and `tensorflow`.

<img width="214" alt="plot" src="https://user-images.githubusercontent.com/36447056/129340614-fdf59b4b-b776-4a95-a614-136c55135f63.png">


## Outlier detection
File: *AE_outlier_detector.ipynb*

In this notebook we use again the AutoEncoder but this time as an outlier detector. The trick here is to train the neural-network on **1 class only**. Indeed, it enables him to learn the structure of the class (e.g. geniune credit card transaction) such that after training, when we feed him with a sample that is from another class, he will struggle to reproduce it. In other words, the reconstruction error will be significantly higher than when he tries to reproduce a sample that comes from the class that he knows.  

To showcase the concept, we use the credit card dataset from Kaggle (download [here](https://www.kaggle.com/mlg-ulb/creditcardfraud)). Note, the dataset is highly imbalanced (0.172% of signal) and we do not rebalance the dataset. After re-scaling the features and training, we geta AUC-ROC of  91% on the test set. The main libraries are `scikit-learn` and `tensorflow`.   

<img width="246" alt="Untitled" src="https://user-images.githubusercontent.com/36447056/129371681-baa520d3-4988-4c5f-994d-ff9fde43b6a2.png">


