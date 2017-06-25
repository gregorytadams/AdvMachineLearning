# home_for_broken_code.py

# Place to dump code I abandonded bu might come back to.  Easier than reverting for small changes.


# def vote(text_data, network_data, list_of_text_classifiers, list_of_network_classifiers):
#   X_train_text, Y_train_text, x_test_text, y_test_text = train_test_split(text_data)
#   X_train_ntwk, Y_train_ntwk, x_test_ntwk, y_test_ntwk = train_test_split(network_data)
#   for clf in list_of_text_classifiers:
#       clf.predict(x_test, y_test)
#   X_train, Y_train, x_test, y_test = train_test_split(X, y)

# def get_predictions(X_test, y_test, list_of_classifiers):
    
#   predictions = []
#   for clf in list_of_classifiers:
#       clf.predict(X_test, y_test)
        
# Integrated 
# def get_preds_from_text(k = K, num_to_grab = NUM_TO_GRAB):
#     '''
#     Gives ROC stats for best classifiers.
#     '''
#     text_data, labels = fetch_and_format()
#     df_text_models = read_in_df_text()
#     network_data, labels = get_network_data()
#     df_network_models = read_in_df_network(NUM_TO_GRAB)

#     all_preds = []
#     network_preds = []
#     text_preds = []


#     for index, row in df_text_models.iterrows():
#         print("Text Model {}".format(index))
#         if index == 2:
#             break # just for testing.
#         AUROC = []
#         for i in range(k):
#             print("Fold {}".format(i))
#             data_amount = TOTAL_NUM_BILLS / (k+1) * (i+1) #the size of the fold in k-fold
#             testing_text_data = []
#             training_text_data = []



#             testing_labels = []
#             training_labels = []
            
#             # testing_bills = []
#             # training_bills = []
            
#             for key in text_data:
#                 # print(key)
#                 if key < data_amount:
#                     training_text_data.append(text_data[key])
#                     training_labels.append(labels[key])
#                     # training_bills.append(key)
#                 else:
#                     testing_text_data.append(text_data[key])
#                     testing_labels.append(labels[key])
#                     # testing_bills.append(key)
#             pl = None # Greg, don't delete this line again.  It is necessary.
#             pl = deepcopy(PL)
#             pl = Pipeline(pl)
#             params = ast.literal_eval(row['params'])
#             pl.set_params(**params)
#             pl.fit(training_text_data, training_labels)
#             # pred_probs = pl.predict_proba(testing_data)
#             # plot_roc(testing_labels, np.array(pred_probs)[:,1], str(index) + '|' + str(i))
#             # AUROC.append(roc_auc_score([1 if i == 'passed' else 0 for i in testing_labels], np.array(pred_probs)[:,1]))
#         all_preds.append(pl.predict(testing_text_data))
#     return all_preds, testing_labels

# Made global variables in text_classifier.py
def define_parameters():
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf_SVM = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier())])
    # The actual
    # parameters_NB = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'vect__binary': (True, False), \
    #             'tfidf__use_idf': (True, False), 'tfidf__norm': ('l1', 'l2', None), 'tfidf__smooth_idf': (True, False), \
    #             'tfidf__sublinear_tf': (True, False), 
    #             'clf__alpha': (10, 1, 1e-1, 1e-2, 1e-3, 1e-4)}

    # For test, so it doesn't take forever... lol
    parameters = {'vect__ngram_range': [(1, 1)], 'vect__binary': (True, False), \
            'tfidf__use_idf': (True, False), 'tfidf__norm': ('l1', None), 'tfidf__smooth_idf': (True, False), \
            'tfidf__sublinear_tf': (True, False), 
            'clf__alpha': (10, 1e-4)}
    return text_clf, parameters 


#rewriteen in main function to network
def get_predictions(node, master_dict, passage_data):
  '''
  Unfinished
  '''

  for bill in range(num_bills_training, len(master_dict)):
      add_node(undirected_people_graph)
      und_cm = general_centrality_measures(undirected_people_graph)
      add_node(directed_people_graph)
      dir_cm = general_centrality_measures()
      sponsor_dir_cms = list(map(lambda x: x[master_dict[bill][0]], dir_cm))
      sponsor_und_cms = list(map(lambda x: x[master_dict[bill][0]], und_cm))

      for i in range(len(dir_cm)):
          l = []
          for cosp in master_dict[bill][1]:

      add_node(bill_graph)


# Modified to make init_dict
def split_master_dict(master_dict, num_bills = 3000):
  '''
  Splits into temporal train/test set.  Turns out it's completely unnecessary.
  '''
  training_dict = {}
  testing_dict = {}
  for key in master_dict:
      if key < num_bills:
          new_dict[key] = master_dict[key]
      else:
          testing_dict[key] = master_dict[key]
  return training_dict, testing_dict




# Combined into one generic function

# def get_master_dict(filename = 'bill_sponsorships_final.json'):
#   '''
#   Just loading up the dict from my web scraper
#   '''
#   with open(filename) as f:
#       master_dict = json.loads(f.read())
#   new_master_dict = {}
#   for key in master_dict: # I should have used pickle, but it doesn't take enough time to justify changing it. Lesson learned.
#       new_master_dict[int(key)] = master_dict[key]
#   return new_master_dict

# def get_passage_data(filename = 'passage_data.json'):
#   '''
#   Gets passage dict.
#   '''
#   with open(filename) as f:
#       passage_dict = json.loads(f.read())
#   new_passage_dict = {}
#   for key in master_dict: # I should have used pickle, but it doesn't take enough time to justify changing it. Lesson learned.
#       new_master_dict[int(key)] = master_dict[key]
#   return new_master_dict


#never built out everything for bill network

def undirected_centrality_measures(graph):
    '''
    I don't ever actually use this (for simplicity's sake), but could be an extension
    '''
    betweenness = nx.betweenness_centrality(graph, weight = "weight", k = 50)
    print('Got General Betweenness')
    current_flow_betweenness = nx.current_flow_betweenness_centrality(graph, weight = "weight")
    print('Got Current Flow Betweenness')
    return [betweenness, current_flow_betweenness]

# from all_centrality_scores -- never built out bill network
    if undirected: # Should never be true; I didn't build out the rest of the program to account for this b/c time
        centrality_measures += undirected_centrality_measures(graph)
        measures += ['betweenness centrality', 'current flow betweenness centrality']




    # print(all_preds)
    # final_preds = mode(all_preds).mode[0]
    # print(final_preds)
    # print(testing_labels[:10], final_preds[:10])
    # print('----------')
    # print(precision_recall_fscore_support(testing_labels, final_preds)[2])


    # print(all_preds)
    # final_preds = mode(all_preds).mode[0]
    # print(final_preds)
    # print(testing_labels[:10], final_preds[:10])
    # print(precision_recall_fscore_support(testing_labels, final_preds))
        # print(np.mean(np.array(AUROC)))

            # build_and_validate(text_data, labels, [(pl, row['params'])])


# def build_and_validate(data, labels, pairs, just_testing = IM_JUST_HERE_TO_TEST):
#     '''
#     Builds and tests the models for each of the classifiers and parameters.

#     Inputs:
#     data, labels; from fetch_and_format
#     just_testing, a bool indicating whether you want real results, or want to not break your computer 

#     Outputs the model evaulation report to a csv.
#     '''
#     for tup in pairs:
#         gs_clf = GridSearchCV(tup[0], tup[1], n_jobs = -1, cv = K) # -1 parallelizes on all available cores
#         if just_testing:
#             gs_clf = gs_clf.fit(data[:100], labels[:100])
#         else:
#             gs_clf = gs_clf.fit(data, labels)
#         for param_name in sorted(tup[1].keys()):
#             print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
#         df = pd.DataFrame(gs_clf.cv_results_)
#         df.to_csv('output_for_pres_output_{}.csv'.format(i))
#         gs_clf = None



# def plot_roc(y_true, y_score, model_name):
#     # print(y_true)
#     # print(y_score)
#     fpr, tpr, _ =  roc_curve(y_true, y_score, pos_label='passed')
#     plt.clf()
#     plt.plot(fpr, tpr, lw=2, color='red', label='ROC curve')
#     plt.plot([0,1], [0,1], color='blue', label='Random Classifier')
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('ROC')
#     plt.savefig('{}ROC{}.png'.format(WHERE_TO_SAVE_PLOTS, model_name))
#     plt.close('all')


# def rebuild_and_get_point_metrics_for_text(X, y, top_models, clfs = CLFS, thresholds = THRESHOLDS, plot = False):
#     '''
#     Rebuilds best models from magic loop and gets point metrics at various thresholds.  Optionally plots.
#     '''
#     model_list = [[]]
#     for thresh in thresholds:
#         model_list[0] += ['F1 at '+str(thresh), 'Precision at ' + str(thresh), \
#                             'Recall at '+str(thresh), 'Accuracy at '+str(thresh)]    
#     for index, row in top_models.iterrows():
#         clf = clfs['NB']
#         p = row['Params']
#         clf.set_params(**p)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
#         clf.fit(X_train, y_train)
#         y_pred_probs = clf.predict_proba(X_test)[:,1]
#         row_of_metrics = []
#         for thresh in thresholds:
#             y_pred = np.asarray([1 if i >= thresh else 0 for i in y_pred_probs])
#             row_of_metrics += get_model_evaluation_stats(y_test, y_pred)
#         if plot:
#             plot_precision_recall(y_test, y_pred_probs, index)
#             plot_roc(y_test, y_pred_probs, index)
#         model_list.append(row_of_metrics)
#     df = pd.DataFrame(model_list[1:], columns = model_list[0]) 
#     print(df)
#     new_df = pd.concat([top_models, df], axis=1)
#     return new_df


# def remove_junk(string):
#     '''
#     Cleans up the text of the bills.
#     '''
#     string = string.lower()
#     remove_digits = string.maketrans('', '', digits)
#     remove_punctuation = string.maketrans('', '', punctuation)
#     remove_newlines = string.maketrans('', '', '\n')
#     remove_indicators = string.maketrans('','','PUBLIC LAW')
#     res = string.translate(remove_digits).translate(remove_punctuation).translate(remove_newlines).translate(remove_indicators)
#     return res
