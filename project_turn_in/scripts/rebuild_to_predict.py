# Integrate and Predict

from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import  roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.pipeline import Pipeline
from os import walk
from string import digits, punctuation
import ast
import pylab as pl
import matplotlib.pyplot as plt
from scipy.stats import mode 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# Things I def need
from text_classifier import remove_junk
from build_classifier import plot_roc


CLFS = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
    'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'LR': LogisticRegression(penalty='l1', C=1e5),
    'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
    'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    # 'SGD': SGDClassifier(loss="hinge", penalty="l2"), #Doesn't give probability estimates or there's an overflow.  SVM is similar anyway.
    'KNN': KNeighborsClassifier(n_neighbors=3) 
    }

SENTIMENT_CATEGORIES = ['passed', 'not_passed']
PL = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('NB', MultinomialNB())]

WHERE_TO_SAVE_PLOTS = './'
TOTAL_NUM_BILLS = 4122
K = 2
NUM_TO_GRAB = 10

def read_in_df_text(num_to_grab = 10, rank_cutoff = 3, filename = '../output/data_outputs/test_output_text_classifier_NB.csv'):
    '''
    Reads in the csv of the top text classifiers
    '''
    df = pd.read_csv(filename)
    df = df[df.rank_test_score >= rank_cutoff]
    if len(df) < num_to_grab:
        print("Warning: There are too few models from your text")
    return df.head(num_to_grab)

def read_in_df_network(num_to_grab = 10, filename = '../output/data_outputs/top_models_from_build_classifier.csv'):
    '''
    Reads in the csv of the top 
    '''
    df = pd.read_csv(filename)
    if len(df) < num_to_grab:
        print("Warning: There are too few models from your network")
    return df.head(num_to_grab)

def get_network_data(filename = '../output/data_outputs/network_with_senators.csv'):
    df = pd.read_csv(filename)
    network_data = {}
    labels = {}
    for i, row in df.iterrows():
        network_data[int(row[1])] = row[4:]
        labels[int(row[1])] = row[3]
    return network_data, labels

# Old Version
def get_preds_from_network(network_data, labels, k = K, num_to_grab = NUM_TO_GRAB, ):
    df_network_models = read_in_df_network(NUM_TO_GRAB)
    all_preds = []
    for index, row in df_network_models.iterrows():
        print("Network Model {}".format(index))
        # AUROC = []
        for i in range(k):
            print("Fold {}".format(i))
            data_amount = TOTAL_NUM_BILLS / (k+1) * (i+1) #the size of the fold in k-fold
            testing_network_data = []
            training_network_data = []
            testing_labels = []
            training_labels = []
            for key in network_data:
                if key < data_amount:
                    training_network_data.append(network_data[key])
                    training_labels.append(labels[key])
                else:
                    testing_network_data.append(network_data[key])
                    testing_labels.append(labels[key])
            params = ast.literal_eval(row['Parameters'])
            model_key = row['Models']
            clf = CLFS[model_key].set_params(**params)
            clf.fit(training_network_data, training_labels)

            # I left this in to give you an easier look inside.

            # pred_probs = clf.predict_proba(testing_data)
            # plot_roc(testing_labels, np.array(pred_probs)[:,1], str(index) + '|' + str(i))
            # AUROC.append(roc_auc_score([1 if i == 'passed' else 0 for i in testing_labels], np.array(pred_probs)[:,1]))
        # print(np.mean(AUROC))

        all_preds.append(clf.predict(testing_network_data))
    return all_preds


def get_preds_from_text(text_data, labels, k = K, num_to_grab = NUM_TO_GRAB):
    '''
    Gives ROC stats for best classifiers.
    '''
    df_text_models = read_in_df_text()
    all_preds = []
    for index, row in df_text_models.iterrows():
        print("Text Model {}".format(index))
        # AUROC = []
        for i in range(k):
            print("Fold {}".format(i))
            data_amount = TOTAL_NUM_BILLS / (k+1) * (i+1) #the size of the fold in k-fold
            testing_text_data = []
            training_text_data = []
            testing_labels = []
            training_labels = []
            for key in text_data:
                if key < data_amount:
                    training_text_data.append(text_data[key])
                    training_labels.append(labels[key])
                else:
                    testing_text_data.append(text_data[key])
                    testing_labels.append(labels[key])
            pl = deepcopy(PL)
            pl = Pipeline(pl)
            params = ast.literal_eval(row['params'])
            pl.set_params(**params)
            pl.fit(training_text_data, training_labels)

            # Left in for testing as well

            # pred_probs = pl.predict_proba(testing_data)
            # plot_roc(testing_labels, np.array(pred_probs)[:,1], str(index) + '|' + str(i))
            # AUROC.append(roc_auc_score([1 if i == 'passed' else 0 for i in testing_labels], np.array(pred_probs)[:,1]))
        # print(np.mean(AUROC))

        all_preds.append(pl.predict(testing_text_data))
    return all_preds, testing_labels


def go():
    '''
    Master Function.
    '''
    text_data, labels_from_text = fetch_and_format()
    network_data, labels_from_network = get_network_data()

    # Deals with missing data.  I apologize for hackiness.
    for key in range(TOTAL_NUM_BILLS+1):
        if key not in text_data and key in network_data:
            del network_data[key]
            del labels_from_network[key]
        if key not in network_data and key in text_data:
            del text_data[key]
            del labels_from_text[key]

    text_preds, testing_labels = get_preds_from_text(text_data, labels_from_text)
    network_preds = get_preds_from_network(network_data, labels_from_network)
    all_preds = text_preds + network_preds 
    final_preds = mode(np.array(all_preds)).mode[0]
    metrics = precision_recall_fscore_support(testing_labels, final_preds)
    print("Precision: {}".format(metrics[0]))
    print("Recall: {}".format(metrics[1]))
    print("F1: {}".format(metrics))



# ===============================================
# =========== HELPER FUNCTIONS ==================
# ===============================================

def fetch_and_format():
    '''
    Fetches and formats the data. 
    Copied and modified from text_classifier.py (I had to change the data structure it returned).

    Outputs:
    data, a dict of bill: text
    labels, a dict of bill:label
    '''
    print("Fetching text data...")
    data = {}
    labels = {}
    counter = 0
    for cat in SENTIMENT_CATEGORIES:
        filenames = next(walk('./all_bills/{}/txts'.format(cat)))[2]
        for f in filenames:
            with open('./all_bills/{}/txts/{}'.format(cat, f)) as txt_f:
                data[int(f[:-4])] = remove_junk(txt_f.read())
                labels[int(f[:-4])] = cat
            counter += 1
            if counter % 1000 == 0:
                print("Got bill {}".format(counter))
    return data, labels

# if __name__ == "__main__":
#     go()


