from numpy import exp, array, random, dot
import numpy as np
#сеть с двумя нейронами
tsi = np.array([   [1, 1, 1, 1, 1, 0],
				   [0, 1, 1, 1, 1, 1],
				   [0, 1, 1, 1, 1, 0]])
weights1 = np.array([ [0.5, 0.5],
					 [0.5, 0.5],
					 [0.5, 0.5],
					 [0.5, 0.5],
					 [0.5, 0.5],
					 [0.5, 0.5]  ])
weights2 = np.array([0.5, 0.5])
inp = np.array([1, 0, 1])
def sig(x):
	return 1 / (1 + exp(-x))
iterat = 0
Loss = 1
while Loss > 0.00001:
#after f1 and f2
	a_f1_f2 = np.dot(weights1.T, tsi[iterat%2].T)
#after sigmoida
	a_sigm1 = sig(a_f1_f2)
#after f3
	a_f3 = weights2 @ a_sigm1
#after second sigmoida
	a_sigm2 = sig(a_f3)
#counting weights
	weights2 += (a_sigm2 - inp[iterat%2]) * (a_sigm2 * (1 - a_sigm2) * a_sigm1) * 1
	weights1 += (np.column_stack([tsi[iterat%2], tsi[iterat%2]]) * ((a_sigm2 - inp[iterat%2]) * a_sigm2 * (1 - a_sigm2) * \
	weights2 * a_sigm1 * (1 - a_sigm1))[None,:]) * 1
	if iterat % 10000==0:
		print('weights1 = ', weights1, '\n', 'weights2 = ', weights2, '\n', 'Loss = ', (inp[iterat%2] - sig(sig(weights1.T @ tsi[iterat%2].T).T @ weights2))**2/2, '\n', \
		'Iteration = ', iterat%2)
	iterat += 1
	Loss = (inp[0] - sig(weights2 @ sig(weights1.T @ tsi[iterat%2].T)))**2/2
print('Result = ', sig(weights2 @ sig(weights1.T @ tsi[iterat%2].T)), '\n', \
	  'Loss = ', Loss, \
	  'Iterat = ', iterat)

