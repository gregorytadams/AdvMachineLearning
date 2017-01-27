# This was where we tried to do page rank and failed.  Code largely is broken; we include it to show some of the efforts we made

import numpy as np
import random
'''
{senator:{relationships:{senator:number}}}

for senator in rep_dict:
	

1. generate matrix
2. sqrt each cell (to flatten out matrix)
3. transition matrix?
'''


def generate_matrix(rep_dict):
	# Assign each senator to a row. 
	# From https://github.com/govtrack/govtrack.us-web/blob/master/analysis/sponsorship_analysis.py line 115
	rep_row_index = {}
	for s_id in rep_dict:
		if s_id not in rep_row_index:
			rep_row_index[s_id] = len(rep_row_index)
	total_reps = len(rep_row_index)
	sponsorship_matrix = np.zeros(total_reps, total_reps)


def normalize2(M):
	for x in range(len(M)):
		M[:,x] = M[:,x]/sum(M[:,x])
	return M

def test():
	v = np.ones((10,1))/10
	x = np.ones((10))

def transitions(sponsorship):
	out_degrees = []
	for i in sponsorship:
		out_degrees.append(sum(i))
	n = len(out_degrees)
	transition_matrix = (0.9*sponsorship/out_degrees) + (0.1*1/n)
	return transition_matrix

def markov_rank(transition_matrix, num_steps):
	return None

def make_one_move(transition_matrix, senator):
	# from CS121 lecture notes on PageRank
	(rows,cols) = transition_matrix.shape()
	r = random.uniform(0.0,1.0)
	psum = 0.0
	for other in range(rows):
		psum += transition_matrix[senator][other]
		if psum >= r:
			return j
	return n

def random_surfer(transition_matrix, num_steps):
	return None
