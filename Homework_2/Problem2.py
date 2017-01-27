import numpy.linalg as npl
import numpy as np

# Note: Please make sure you're using python3.  Python2.x returns 0 for 
# the integer division.
M = np.array([[0.0, .5, 1.0, 0.0], [1/3, 0, 0, 0.5], \
              [1/3, 0, 0, 0.5], [1/3, 0.5, 0, 0]])
X0 = np.array([0.25, 0.25, 0.25, 0.25])

def go(transition_matrix, initial_probability_vector, limit):
    for i in range(1,limit+1):
        transition_matrix = transition_matrix.dot(transition_matrix)
        # Below is useful to see the returned vector "bottom out"
        # print(transition_matrix.dot(initial_probability_vector))
    return transition_matrix.dot(initial_probability_vector)

if __name__ == "__main__":
  print(go(M, X0, 25))
