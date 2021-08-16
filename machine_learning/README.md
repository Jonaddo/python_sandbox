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

The dataset for this little tutorial can be downloaded on Kaggle [here](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset).  Since we are dealing with health data, in particular whether or not a patient can have a heart attack, we will prefer to chose a model that has a bigger number of false positive (FP) cases than false negative (FN). Indeed the consequences associated by predicting a patient to likely not have a heart attack (but that will in fact have one) is much greater than the opposite. Here below the confusion matrix shows that we have more FP (4) than FN (3).  
  
![KNN_conf_matrix](https://user-images.githubusercontent.com/36447056/129595585-09e55e69-a4e2-43f7-909b-3a02787b2767.png)


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
Indeed after training, we can use the *encoder* part of the **AE** to reduce our input dimension. In the first part of the notebook we reduce the input dimension from **13 to 2 features** then in the second part we also show how to use the latent space aka compressed information as *new* input for a classifier.

In order to showcase the concept, we use the heart attack dataset (download from Kaggle [here](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)) and a random forest in the second part for classification. We use as template the autoencoder code from [tensorflow.org](https://www.tensorflow.org/tutorials/generative/autoencoder) and adjust it to our use case.  
The main libraries are `scikit-learn` and `tensorflow`.  

**Note**: in practice we would probably prefer to not reduce dimension since its a rather very small dataset!

![roc_curves_comparison](https://user-images.githubusercontent.com/36447056/129444676-96a819ea-2044-4056-9c49-a4a56dbf9d66.png)


## Outlier detection
File: *AE_outlier_detector.ipynb*

In this notebook we use again the AutoEncoder but this time as an outlier detector. The trick here is to train the neural-network on **1 class only**. Indeed, it enables him to learn the structure of the class (e.g. geniune credit card transaction) such that after training, when we feed him with a sample that is from another class, he will struggle to reproduce it. In other words, the reconstruction error will be significantly higher than when he tries to reproduce a sample that comes from the class that he knows.  

To showcase the concept, we use the credit card dataset from Kaggle (download [here](https://www.kaggle.com/mlg-ulb/creditcardfraud)). Note, the dataset is highly imbalanced (0.172% of signal) and we do not rebalance the dataset. After re-scaling two features and training, we get a AUC-ROC of 89% on the test set. The main libraries are `scikit-learn` and `tensorflow` and we use as template the autoencoder code from [tensorflow.org](https://www.tensorflow.org/tutorials/generative/autoencoder) and adjust it to our use case.   
  
![AE_outlier_roc_curve](https://user-images.githubusercontent.com/36447056/129600902-61fc6058-fa37-46ce-8a58-82b6938b2a11.png)




