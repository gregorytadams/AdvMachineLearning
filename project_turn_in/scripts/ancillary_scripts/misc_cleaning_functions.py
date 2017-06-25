# misc_cleaning_functions.py
# These are just functions that are a bit too long to do in ipython
# Many used only once to clean, modify, etc.
# Generally safe to ignore; just including them in case they're helpful.

import os
import re
import pandas as pd
import numpy as np
from os import walk
from string import digits, punctuation

def get_passage_data(folder):
    '''
    This function does 3 things:
    (1) gets a dict of whether or not each specific bill passed 
    (2) renames everything to make it easier to find each bill by number
    (3) removes duplicate bills 

    '''
    os.chdir(folder)
    passage_dict = {}
    primary_regex = re.compile('s[0-9]{1,4}')
    secondary_regex = re.compile('S. [0-9]{1,4}')
    for fold in next(os.walk('.'))[1]:
        os.chdir('{}/txts/'.format(fold))
        for f in next(os.walk('.'))[2]:
            matches = primary_regex.findall(f)
            if matches == []:
                with open(f) as f2:
                    name = secondary_regex.findall(f2.read())[0][3:]
            else:
                name = matches[0][1:]
            passage_dict[int(name)] = fold
            name = '{}.txt'.format(name)
            # os.rename(f, name)
            print(f)
            print(name)
        os.chdir('../..')
    os.chdir('..')
    return passage_dict


def merge_milestones(folder_with_dicts):
    '''
    Merges milestones from get_all_sponsorships.  Not used in go function, but useful nonetheless.

    inputs:
    folder_with_dicts, the filepath to the folder that contains nothing but the milestone jsons

    outputs:
    saves a file with the merged json.

    Congress.gov started rate-limiting me (I was following the robots.txt instructions, but ok w/e).  To get around this, I 
    had to make sure to save my progress whenever I ran the scraper, so this merges my dicts and creates the general file.
    '''
    master_dict = {}
    for json_file in next(walk(folder_with_dicts))[2]:
        d = loads(folder_with_dicts + json_file)
        master_dict.update(d)
    with open('bill_sponsorships.json', 'w') as f:
        dump(master_dict, f)


def get_all_senators(master_dict):
    '''
    Gets list of each senator from master_dict.
    dict structure: bill_name:[senator1, [senator2, senator3, ...]]
    '''
    senators = set()
    for val in master_dict.values():
        senators.add(val[0])
        for i in val[1]: senators.add(i)
    return list(senators)

def get_top_senators(network_df_filename):
    df = pd.read_csv(network_df_filename)
    d = {}
    for name in df['Sponsor'].unique():
        d[name] = np.mean(df['dir_pagerank_sponsor'][df.Sponsor == name])
    # for i in sorted(d, key=d.__getitem__):
    #     (i, d[i])
    return d


def fetch_and_format():
    '''
    Fetches and formats the data. Assumes polarity data was extracted in the current folder.  
    Hardcoded for polarity dataset.

    Outputs:
    data, a list of raw strings from the text files
    labels, a list of strings of the SENTIMENT_CATEGORIES, corresponding to the data's labels
    '''
    print("Fetching data...")
    data_passed = ''
    data_failed = ''
    counter = 0
    for cat in ['passed', 'not_passed']:
        filenames = next(walk('./all_bills/{}/txts'.format(cat)))[2]
        for f in filenames:
            with open('./all_bills/{}/txts/{}'.format(cat, f)) as txt_f:
                if cat == 'passed':
                    data_passed += remove_junk(txt_f.read())
                elif cat == 'not_passed':
                    data_failed += remove_junk(txt_f.read())
                # labels.append(cat)
            counter += 1
            if counter % 100 == 0:
                print("{}/4502".format(counter))
    return data_passed.split(' '), data_failed.split(' ')

def count_words(data):
    d = {}
    for i in data:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d

def get_top_words(data_passed, data_failed):
    '''
    '''
    data_passed = count_words(data_passed)
    data_failed = count_words(data_failed)
    sorted_data_passed = []
    sorted_data_failed = []
    for i in sorted(data_passed, key=data_passed.__getitem__):
        sorted_data_passed.append(i)
    for i in sorted(data_failed, key=data_failed.__getitem__):
        sorted_data_failed.append(i)
    top_words_passed = []
    i = 1
    while len(top_words_passed) < 15:
        if sorted_data_passed[i*-1] not in sorted_data_failed[-200:] and len(sorted_data_passed[i*-1]) > 3:
            top_words_passed.append(sorted_data_passed[i*-1])
        i+=1
    i = 1
    top_words_failed = []
    while len(top_words_failed) < 15:
        if sorted_data_failed[i*-1] not in sorted_data_passed[-200:] and len(sorted_data_failed[i*-1]) > 3:
            top_words_failed.append(sorted_data_failed[i*-1])
        i+=1
    return top_words_passed, top_words_failed
