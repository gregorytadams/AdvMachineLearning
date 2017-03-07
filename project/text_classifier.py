# Greg Adams
# 
# Note: I borrowed heavily (incl. minimal copy/paste of individual lines of code) from the sklearn 
# tutorial found at http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# http://stats.stackexchange.com/questions/158027/how-do-i-improve-the-accuracy-of-my-supervised-document-classification-model


print("Importing...")
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, auc, \
                            roc_curve, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from os import walk
from string import digits, punctuation
import pandas as pd
from copy import deepcopy
print("Finished importing")

K = 2 # This is used for both cross-validation and bootstrapping
SENTIMENT_CATEGORIES = ['passed', 'not_passed']
PL = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), None]

CLFS = {'NB': ('clf', MultinomialNB()), \
        # 'SGD': ('clf', SGDClassifier()), \
        # 'SVM': ('clf', svm.SVC())
}

# For REALSIES

# CLF_PARAMS = {'NB': {'clf__alpha': (10, 1, 1e-1, 1e-2, 1e-3, 1e-4)}, \
#               'SGD': { 'clf__loss': ['hinge','log','perceptron'], 'clf__penalty': ['l2','l1','elasticnet'], 'clf__class_weight': ['balanced', None]}, \
#               'SVM': {'clf__C' :[0.00001, 0.0001, 0.001,0.01,0.1,1,10],'clf__kernel':['linear', 'poly', 'rbf', 'sigmoid']}
#             }

# BASIC_PARAMS = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'vect__binary': (True, False), \
#                 'tfidf__use_idf': (True, False), 'tfidf__norm': ('l1', 'l2', None), 'tfidf__smooth_idf': (True, False), \
#                 'tfidf__sublinear_tf': (True, False)}

# For TESTS

CLF_PARAMS = {'NB': {'clf__alpha': (10, 1e-4)}, \
              # 'SGD': { 'clf__loss': ['hinge'], 'clf__penalty': ['elasticnet'], 'clf__class_weight': ['balanced', None]}, \
              # 'SVM': {'clf__C' :[0.00001, 10],'clf__kernel':['poly', 'sigmoid']}
            }

BASIC_PARAMS = {'vect__ngram_range': [(1, 1), (1,3)], 'vect__binary': (True, False), \
                'tfidf__use_idf': (True, False), 'tfidf__norm': ('l1', None), 'tfidf__smooth_idf': (True, False), \
                'tfidf__sublinear_tf': (True, False)}

def define_parameters(pl = PL, clfs = CLFS, clf_params = CLF_PARAMS, basic_params = BASIC_PARAMS):
    '''
    returns list of tuples [(pipeline, parameters), ...]
    '''
    list_of_pairs = []
    for model in clfs:
        pl[2] = clfs[model]
        new_params = deepcopy(basic_params)
        # print(new_params is basic_params)
        # print(new_params == basic_params)
        # print(basic_params == BASIC_PARAMS)
        # print(basic_params is BASIC_PARAMS)
        # for key in new_params:
        #     print("Key: {}".format(key))
        #     print("Params: {}".format(new_params[key]))
        # print(clf_params[model])
        new_params.update(clf_params[model]) # there's apparently no better way to merge 2 dicts.  Wierd...
        # print(new_params == basic_params)
        # print(new_params)
        list_of_pairs.append((Pipeline(pl), new_params))
    return list_of_pairs




# def define_parameters():
#     text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
#     text_clf_SVM = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier())])
#     # The actual
#     # parameters_NB = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'vect__binary': (True, False), \
#     #             'tfidf__use_idf': (True, False), 'tfidf__norm': ('l1', 'l2', None), 'tfidf__smooth_idf': (True, False), \
#     #             'tfidf__sublinear_tf': (True, False), 
#     #             'clf__alpha': (10, 1, 1e-1, 1e-2, 1e-3, 1e-4)}

#     # For test, so it doesn't take forever... lol
#     parameters = {'vect__ngram_range': [(1, 1)], 'vect__binary': (True, False), \
#             'tfidf__use_idf': (True, False), 'tfidf__norm': ('l1', None), 'tfidf__smooth_idf': (True, False), \
#             'tfidf__sublinear_tf': (True, False), 
#             'clf__alpha': (10, 1e-4)}
#     return text_clf, parameters 


def build_and_validate(data, labels):
    # i=0
    pairs = define_parameters()
    # print(pairs)
    # print("========================================")
    # for i in pairs:
    #     print(i[0])
    #     print('--------------------------------------------------')
    #     print(i[1])
    #     print('===================================================')
    for tup in pairs:
        print(tup[0])
        print("-------------------------------------------")
        print(tup[1])
        print("=============================================")
        gs_clf = GridSearchCV(tup[0], tup[1], n_jobs = -1, cv = K) # -1 parallelizes on all available cores
        gs_clf = gs_clf.fit(data[:400], data[:400])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        df = pd.DataFrame(gs_clf.cv_results_)
        df.to_csv('test_output_{}.csv'.format(text_clf))
        gs_clf = None


def remove_junk(string):
    strimg = string.lower()
    remove_digits = string.maketrans('', '', digits)
    remove_punctuation = string.maketrans('', '', punctuation)
    remove_newlines = string.maketrans('', '', '\n')
    res = string.translate(remove_digits).translate(remove_punctuation).translate(remove_newlines)
    return res


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
    counter = 0
    for cat in SENTIMENT_CATEGORIES:
        filenames = next(walk('./all_bills/{}/txts'.format(cat)))[2]
        for f in filenames:
            with open('./all_bills/{}/txts/{}'.format(cat, f)) as txt_f:
                data.append(remove_junk(txt_f.read()))
                labels.append(cat)
            counter += 1
            if counter % 100 == 0:
                print("{}/4502".format(counter))
    return data, labels


def go():
    '''
    Master function. Gathers and formats data, builds models and outputs metrics to be reported.
    '''
    data, labels = fetch_and_format()
    counts, tfidf = build_and_validate(data, labels)
    return sum(counts)/len(counts), sum(tfidf)/len(tfidf)

if __name__ == "__main__":
    f1_counts, f1_tfidf = go()
    print("F1 score for counts: {}".format(f1_counts))
    print("F1 score for tfidf: {}".format(f1_tfidf))