#setup_ipython.py

import pandas as pd 
import numpy as np
import sys
sys.path.insert(0, 'C:~/AdvMachineLearning/')
import my_library as mine

def go():
	df = pd.read_csv('cleaned_normal_data.csv')
	df = df[['c_charge_degree', 'sex', 'race', 'age_cat', 'score_text']]
	df = df[df.score_text != "Medium"]
	df = mine.numberize(df, 'c_charge_degree')
	df = mine.numberize(df, 'score_text')
	df = mine.numberize(df, 'sex')
	df = mine.split_to_indicators(df, 'race')
	df = mine.split_to_indicators(df, 'age_cat')
	x = df.drop('score_text', 1)
	y = df['score_text']
	return x, y, df



