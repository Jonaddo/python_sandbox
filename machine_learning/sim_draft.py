import random

import numpy as np
import pandas as pd
from sklearn import datasets
from PyFastBDT import FastBDT
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

random.seed(123)                                    # for reproductibility

N_sim = 50                                           # Number of toy data-sets
random_seed = np.random.randint(N_sim, size=N_sim)

ZERO_INIT = np.zeros((N_sim, 1))

sim_lr_acc = ZERO_INIT
sim_lr_aucroc = ZERO_INIT
sim_lr_aucpr = ZERO_INIT
sim_lr_sensitivity = ZERO_INIT
sim_lr_specificity = ZERO_INIT

sim_rf_acc = ZERO_INIT
sim_rf_aucroc = ZERO_INIT
sim_rf_aucpr = ZERO_INIT
sim_rf_sensitivity = ZERO_INIT
sim_rf_specificity = ZERO_INIT

sim_bdt_acc = ZERO_INIT
sim_bdt_aucroc = ZERO_INIT
sim_bdt_aucpr = ZERO_INIT
sim_bdt_sensitivity = ZERO_INIT
sim_bdt_specificity = ZERO_INIT


if __name__ == '__main__':
    for i in range(N_sim):
        # Data-set creation
        X, y = datasets.make_classification(
		n_samples=100000,
		n_features=20,
		n_informative=2,
		weights=[0.995, 0.005],
        	n_redundant=2,
		random_state=random_seed[i]
	)

        # Data pre-processing
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_seed[i])

        # LOGISTIC REGRESSION
        logreg = LogisticRegression(C=1, solver='lbfgs', n_jobs=-1)  # set n_jobs=1 to use only 1 CPU and -1 for all
        logreg.fit(X_train, y_train)

        y_test_lr_predProba = logreg.predict_proba(X_test)[:, 1]
        y_test_lr_pred = logreg.predict(X_test)

        sim_lr_acc[i, 0] = accuracy_score(y_test, y_test_lr_pred)                                          # Accuracy
        sim_lr_aucroc[i, 0] = roc_auc_score(y_test, y_test_lr_predProba)                                   # AUC-ROC
        sim_lr_precisionCurve, sim_lr_recallCurve, _ = precision_recall_curve(y_test, y_test_lr_predProba, pos_label=1)
        sim_lr_aucpr[i, 0] = auc(sim_lr_recallCurve, sim_lr_precisionCurve, reorder=False)                 # AUC-PR
        sim_lr_cm = confusion_matrix(y_test, y_test_lr_pred)
        sim_lr_sensitivity[i, 0] = sim_lr_cm[0, 0] / (sim_lr_cm[0, 0] + sim_lr_cm[0, 1])                   # Sensitivity
        sim_lr_specificity[i, 0] = sim_lr_cm[1, 1] / (sim_lr_cm[1, 0] + sim_lr_cm[1, 1])                   # Specificity

        # RANDOM FOREST
        randforest = RandomForestClassifier(n_estimators=100, max_depth=3, n_jobs=-1)  # set n_jobs=1 to use only 1 CPU
        randforest.fit(X_train, y_train)

        y_test_rf_predProba = randforest.predict_proba(X_test)[:, 1]
        y_test_rf_pred = randforest.predict(X_test)

        sim_rf_acc[i, 0] = accuracy_score(y_test, y_test_rf_pred)                                          # Accuracy
        sim_rf_aucroc[i, 0] = roc_auc_score(y_test, y_test_rf_predProba)                                   # AUC-ROC
        sim_rf_precisionCurve, sim_rf_recallCurve, _ = precision_recall_curve(y_test, y_test_rf_predProba, pos_label=1)
        sim_rf_aucpr[i, 0] = auc(sim_rf_recallCurve, sim_rf_precisionCurve, reorder=False)                 # AUC-PR
        sim_rf_cm = confusion_matrix(y_test, y_test_rf_pred)
        sim_rf_sensitivity[i, 0] = sim_rf_cm[0, 0] / (sim_rf_cm[0, 0] + sim_rf_cm[0, 1])                   # Sensitivity
        sim_rf_specificity[i, 0] = sim_rf_cm[1, 1] / (sim_rf_cm[1, 0] + sim_rf_cm[1, 1])                   # Specificity

	# FASTBDT
        clf = FastBDT.Classifier(nTrees=100, depth=3)
        clf.fit(X=X_train, y=y_train)

        p = clf.predict(X_test)
        y_pred = [1 if e > 0.5 else 0 for e in p]

        sim_bdt_acc[i, 0] = accuracy_score(y_test, y_pred)  												# Accuracy
        sim_bdt_aucroc[i, 0] = roc_auc_score(y_test, p)  													# AUC-ROC
        sim_bdt_precisionCurve, sim_bdt_recallCurve, _ = precision_recall_curve(y_test, p, pos_label=1)
        sim_bdt_aucpr[i, 0] = auc(sim_bdt_recallCurve, sim_bdt_precisionCurve, reorder=False)  				# AUC-PR
        sim_bdt_cm = confusion_matrix(y_test, y_pred)
        sim_bdt_sensitivity[i, 0] = sim_bdt_cm[0, 0] / (sim_bdt_cm[0, 0] + sim_bdt_cm[0, 1])  				# Sensitivity
        sim_bdt_specificity[i, 0] = sim_bdt_cm[1, 1] / (sim_bdt_cm[1, 0] + sim_bdt_cm[1, 1]) 				# Specificity

        print('Simulation:', i+1, 'sur', N_sim)
    print('Done')

# Print the description results
COLS = ["AUC-PR", "AUC-roc", "Accuracy", "Sensitivity", "Specificity"]

df_sim_lr = pd.DataFrame(np.c_[sim_lr_aucpr, sim_lr_aucroc, sim_lr_acc, sim_lr_sensitivity, sim_lr_specificity])
df_sim_lr = pd.DataFrame(df_sim_lr.values, columns=COLS)
print('----- Logistic Regression -----')
print(df_sim_lr.describe())

df_sim_rf = pd.DataFrame(np.c_[sim_rf_aucpr, sim_rf_aucroc, sim_rf_acc, sim_rf_sensitivity, sim_rf_specificity])
df_sim_rf = pd.DataFrame(df_sim_rf.values, columns=COLS)
print('----- Random Forest -----')
print(df_sim_rf.describe())

df_sim_bdt = pd.DataFrame(np.c_[sim_bdt_aucpr, sim_bdt_aucroc, sim_bdt_acc, sim_bdt_sensitivity, sim_bdt_specificity])
df_sim_bdt = pd.DataFrame(df_sim_bdt.values, columns=COLS)
print('----- FastBDT -----')
print(df_sim_bdt.describe())
