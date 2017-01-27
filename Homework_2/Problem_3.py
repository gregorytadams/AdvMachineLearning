# Greg Adams
# 
# Note: I borrowed heavily (incl. minimal copy/paste of individual lines of code) from the sklearn 
# tutorial found at http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


print("Importing...")
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel
import os

SENTIMENT_CATEGORIES = ['pos', 'neg'] # index zero is pos_label for evaluation metrics
K = 15 # This is used for both cross-validation and bootstrapping
VERBOSE = False

def fetch_and_format():
    '''
    Fetches and formats the data. Assumes polarity data was extracted in the current folder.  
    Hardcoded for polarity dataset.

    Outputs:
    data, a list of raw strings from the text files
    labels, a list of strings of the SENTIMENT_CATEGORIES, corresponding to the data's labels
    '''
    print("Fetching data...")
    data = []
    labels = []
    for cat in SENTIMENT_CATEGORIES:
        filenames = next(os.walk('./review_polarity/txt_sentoken/{}'.format(cat)))[2]
        for f in filenames:
            with open('./review_polarity/txt_sentoken/{}/{}'.format(cat, f)) as txt_f:
                data.append(txt_f.read())
                labels.append(cat)
    return data, labels

def get_k_folds(data_list, target_list, k):
    '''
    Gets the k folds for cross validation.

    Inputs:
    data_list, a list of lists, each sublist of which is a sentence
    target_list, a list of strings, each element of which is the label for data_list

    Outputs:
    (all), a list of k folds, each of which is a list of lists with each sublist containing a sentence
    '''
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    # kf.split returns tuple: ([all indices of training data], [all indices of test data])
    for tup in  kf.split(data_list):
        train_data.append([data_list[i] for i in tup[0]]) 
        train_labels.append([target_list[i] for i in tup[0]]) 
        test_data.append([data_list[i] for i in tup[1]]) 
        test_labels.append([target_list[i] for i in tup[1]]) 
    return train_data, train_labels, test_data, test_labels

def get_model_evaluation_stats(y_true, y_pred, labels=None, pos_label=1, average='binary', \
                               sample_weight=None, title="Summary"):
    '''
    Custom summary function to get a quick look at the models.

    Requires import above: 'from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score'
    
    Inputs:
    (all) same inputs as sklearn.model_selection inputs of the same name.  Refer to sklearn documentation at 
    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    '''
    print("TITLE: {}".format(title))
    print("F1: {}".format(f1_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)))
    print("Precision: {}".format(precision_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)))
    print("Recall: {}".format(recall_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)))
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred, sample_weight=sample_weight)))
    print("-"*30)

def build_and_validate(train_data, train_labels, test_data, test_labels):
    '''
    Workhorse of the program.  Takes the data, builds the models, and outputs evaluation metrics.

    Inputs
    (all): outputs from get_k_folds, a list of k folds, each of which is a list of lists with \
    each sublist containing a sentence

    Outputs
    (all): a list of f1 scores from each of K folds of the respective models. 
    len(counts) = len(tfidf) = K

    Note: Borrows heavily from sklearn tutorial linked at top of file.
    '''
    print("Building models...")
    counts = []
    tfidf = []
    for i in range(len(train_data)): 
        # Tokenizes and vectorizes (english: cleans up/formats data)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(train_data[i])

        # trains counts classifer
        clf = MultinomialNB().fit(X_train_counts, train_labels[i])
        X_new_counts = count_vect.transform(test_data[i])

        # predicts for counts classifier
        pred_labels = clf.predict(X_new_counts)
        counts.append(f1_score(test_labels[i], pred_labels, pos_label=SENTIMENT_CATEGORIES[0]))
        if VERBOSE:
            get_model_evaluation_stats(test_labels[i], pred_labels, pos_label=SENTIMENT_CATEGORIES[0], title="Counts")

        # converts data to TF-IDF
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        # trains tfidf classifier
        clf = MultinomialNB().fit(X_train_tfidf, train_labels[i])
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        # predicts for tfidf classifier
        pred_labels = clf.predict(X_new_tfidf)
        tfidf.append(f1_score(test_labels[i], pred_labels, pos_label=SENTIMENT_CATEGORIES[0]))
        if VERBOSE:
            get_model_evaluation_stats(test_labels[i], pred_labels, pos_label=SENTIMENT_CATEGORIES[0], title="TFIDF")

    return counts, tfidf


def go():
    '''
    Master function. Gathers and formats data, builds models and outputs metrics to be reported.
    '''
    data, labels = fetch_and_format()
    train_data, train_labels, test_data, test_labels = get_k_folds(data, labels, K)
    counts, tfidf = build_and_validate(train_data, train_labels, test_data, test_labels)
    p_value = ttest_rel(counts, tfidf)[1] # two tailed independent t-test
    return sum(counts)/len(counts), sum(tfidf)/len(tfidf), p_value

if __name__ == "__main__":
    f1_counts, f1_tfidf, p_value = go()
    print("F1 score for counts: {}".format(f1_counts))
    print("F1 score for tfidf: {}".format(f1_tfidf))
    print("P_value from paired t-test: {}".format(p_value))