# HMM.py
# Implements the HMM class for Homework 3, Problem 3

# Note: I got some help from a friend (another undergrad), Sofia Wyetzner, who has implemented HMMs in the past. 

import numpy as np
from random import randint


FIRST_TEST_A = np.array([[0.0, 0.0, 0.0, 0.0], \
						 [0.8, 0.6, 0.4, 0.0], \
						 [0.2, 0.3, 0.5, 0.0], \
						 [0.0, 0.1, 0.1, 0.0]])
# 						  S    H    C    E	

#						  H    C
FIRST_TEST_B = np.array([[0.2, 0.5], \
						 [0.4, 0.4], \
						 [0.4, 0.1]])

FIRST_TEST_OBS = [3, 3, 3]
SECOND_TEST_OBS = [randint(1,3) for i in range(100)]

class HMM(object):

	def __init__(self, transmission_prob, emission_prob):
		self.transmission_prob = transmission_prob
		self.emission_prob = emission_prob
		self.num_states = len(transmission_prob) - 2


	def train(self, observations):
		# given observations, idk dude
		pass

	def _get_not_quite_chi(observations, i, j, t):
		'''
		i and j are states
		t is the index of observations we are at
		'''
		fwd = _get_forward_likelihood(observations[:t])
		back = _get_backward_likelihood(observations[t+1:])
		aij = self.transmission_prob[i+1][j+1]
		bij = self.emission_prob[observations[t]][j]
		return fwd * back * aij * bij

	def _get_chi(all_observations, sub_observations, i, j, t):
		return self._get_not_quite_chi(sub_observations, i, j, t) / self._get_forward_likelihood(all_observations)

	def _get_forward_likelihood(self, observations):
		'''
		gets likelihood set of observations was produced given a model
		'''
		trellis = np.array([ [0.0]*len(observations) for i in range(self.num_states)])
		# print(trellis)
		for obs_num, obs in enumerate(observations):
			if obs_num == 0:
				for current_state in range(self.num_states):
					trellis[current_state][obs_num] = self.transmission_prob[current_state + 1][0] * \
													  self.emission_prob[obs-1][current_state]
				continue
			for current_state in range(self.num_states): #should be 0 or 1; higher in generalized scenario
				for prev_state in range(self.num_states): #should be either 0 or 1; higher in generalized scenario
					trellis[current_state][obs_num] += trellis[prev_state][obs_num - 1] * \
										   				self.transmission_prob[current_state + 1][prev_state + 1] * \
										   				self.emission_prob[obs-1][current_state]
		end = 0
		for last_value in range(self.num_states):
			end += trellis[last_value][len(observations) - 1] * \
				   self.transmission_prob[self.num_states + 1][last_value+1]
		return end

	def _get_backward_likelihood(self, observations):
		'''
		also gets likelihood set of observations was produced given a model but... differently.
		'''
		trellis = np.array([ [0.0]*len(observations) for i in range(self.num_states)])
		# print(trellis)
		for obs_num, obs in enumerate(observations):
			if obs_num == 0:
				for current_state in range(self.num_states):
					trellis[current_state][obs_num] = self.transmission_prob[self.num_states + 1][current_state + 1] * \
													  self.emission_prob[obs-1][current_state]
				continue
			for current_state in range(self.num_states): #should be 0 or 1; higher in generalized scenario
				for prev_state in range(self.num_states): #should be either 0 or 1; higher in generalized scenario
					trellis[current_state][obs_num] += trellis[prev_state][obs_num - 1] * \
										   				self.transmission_prob[prev_state + 1][current_state + 1] * \
										   				self.emission_prob[obs-1][current_state]
		end = 0
		for last_value in range(self.num_states):
			end += trellis[last_value][len(observations) - 1] * \
				   self.transmission_prob[self.num_states + 1][last_value+1]
		return end

	def _viterbi(self, observations):
		'''
		gets most likely sequence of hidden states (e.g. weather) given observations and a model
		'''
		trellis = np.array([ [0.0]*len(observations) for i in range(self.num_states)])
		path = []
		# print(trellis)
		for obs_num, obs in enumerate(observations):
			if obs_num == 0:
				for current_state in range(self.num_states):
					trellis[current_state][obs_num] = self.transmission_prob[current_state + 1][0] * \
													  self.emission_prob[obs-1][current_state]
				continue
			for current_state in range(self.num_states): #should be 0 or 1; higher in generalized scenario
				l = []
				for prev_state in range(self.num_states): #should be either 0 or 1; higher in generalized scenario
					l.append(trellis[prev_state][obs_num - 1] * self.transmission_prob[current_state + 1][prev_state + 1] * self.emission_prob[obs-1][current_state])
					trellis[current_state][obs_num] = max(l)
				path.append(np.argmax(trellis[:,current_state]))
		return path

	def likelihood(self, observations):
		return self._get_forward_likelihood(observations)
		# vit = self._viterbi(observations)
		# back = self._get_backward_likelihood(observations)






def go():
	my_model = HMM(FIRST_TEST_A, FIRST_TEST_B)
	l = my_model.likelihood(FIRST_TEST_OBS)
	print(l)

if __name__ == "__main__":
	go()