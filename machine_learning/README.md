# About the repository
Here we compare the performance of two sklearn tools (logistic regression (LR) and random forest (RF)) and an enhanced stochastic-gradient boosted decision tree developped by Dr. T. Keck called FastBDT for multivariate classification. In his [paper](https://arxiv.org/abs/1609.06119) he shows that the FastBDT is better than TMVA and XGBoost in terms of runtime and classification quality.
Here is the link to his library [FastBDT](https://github.com/thomaskeck/FastBDT).

In order to make a fair comparaison the FastBDT and the Random forest have both number of trees set to 100, max-depth=3 and the rest is left by default.

# Dependencies
- FastBDT
- numpy
- pandas
- sklearn

# Input dataset
You can download the dataset that I have used for this project at: https://www.kaggle.com/mlg-ulb/creditcardfraud

"The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions." (source: description from the mentioned webpage)

### Data exploration
Here is a very brief visualization decription of the data-set. On the this plot below red crosses are fraud transactions and in bleu genuine transactions.

![timeseries](https://user-images.githubusercontent.com/36447056/39690478-910e10da-51da-11e8-86cc-45c9ed49eeeb.png)

# Simple visual comparison
"A comparison of a several classifiers in scikit-learn on synthetic datasets. The point of this example is to illustrate the nature of decision boundaries of different classifiers. This should be taken with a grain of salt, as the intuition conveyed by these examples does not necessarily carry over to real datasets." Code taken from [here](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)

![sepa_3_roc](https://user-images.githubusercontent.com/36447056/39691042-78f0063c-51dc-11e8-9fad-e0a0e254f88c.png)


# Classifier stress test overview
The following tables shows the details results for 50 toy data-sets with 0.5% of signals (=positive class).  
The FastBDT has in average better performance than the two others in terms of AUC-ROC and AUC-PR. 

<table>
<tr><th>Logistic regression </th><th>Random Forest </th><th>FastBDT </th></tr>
<tr><td>

|      | AUC-PR  | AUC-ROC |              
| ---- |-------- | --------|
| **mean** | 0.2331  | 0.7265  |
| **std**  | 0.1169  | 0.0221  |
| **min**  | 0.0372  | 0.6744  |
| **25%**  | 0.1394  | 0.7134  |
| **50%**  | 0.2399  | 0.7243  |
| **75%**  | 0.3049  | 0.7423  |
| **max**  | 0.4960  | 0.7690  |

</td><td>

|      | AUC-PR  | AUC-ROC |              
| ---- |-------- | --------|
| **mean** | 0.2741  | 0.7212  |
| **std**  | 0.1066  | 0.0196  |
| **min**  | 0.0546  | 0.6514  |
| **25%**  | 0.1991  | 0.7110  |
| **50%**  | 0.2758  | 0.7220  |
| **75%**  | 0.3417  | 0.7335  |
| **max**  | 0.4833  | 0.7675  |

</td><td>

|      | AUC-PR  | AUC-ROC |              
| ---- |-------- | --------|
| **mean** | 0.2931  | 0.7324  |
| **std**  | 0.1204  | 0.0216  |
| **min**  | 0.0548  | 0.6937  |
| **25%**  | 0.2182  | 0.7161  |
| **50%**  | 0.2966  | 0.7339  |
| **75%**  | 0.3842  | 0.7491  |
| **max**  | 0.5067  | 0.7744  |

</td></tr> </table>

### Figure 1: Performance 
Here we can have a look at the average performance for the three methods over different proportion of positive class (%).  
For each point, we apply a 10-Fold CV.  
These synthetic have 14 features, 10'000 observation points and a class imbalance ranging from 0.5% to 50%.

![perf](https://user-images.githubusercontent.com/36447056/125334163-57cc6f00-e34b-11eb-8ca3-bd280a52d5a5.png)


### Figure 2: Performance difference
![perfdelta](https://user-images.githubusercontent.com/36447056/125334204-62870400-e34b-11eb-8c42-c21d3432ddc2.png)  

As expected the BDT seems to overperform overall.  
However the biggest performance delta is obtained by the BDT in highly imbalanced datasets.  
As the datasets become more and more balanced, the 3 algorithms seems to converge to the same performance.  
Note, in order to perform a more rigorous analysis, we should think about which standardisation to use, create more complex training/test datasets and more importantly tune the hyperparameters prior to testing.


