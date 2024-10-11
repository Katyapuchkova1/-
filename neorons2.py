from numpy import exp, array, random, dot
import numpy as np
#сеть с двумя нейронами
tsi = np.array([   [1, 1, 1, 1, 1, 0],
				   [1, 0, 1, 1, 1, 1],
                   [1, 1, 0, 0, 0, 0],
                   [0, 1, 1, 0, 1, 1]     ])
weights1 = np.array([ [0.5, 0.5],
					 [0.5, 0.5],
					 [0.5, 0.5],
					 [0.5, 0.5],
					 [0.5, 0.5],
					 [0.5, 0.5]  ])
weights2 = np.array([0.5, 0.5])
inp = np.array([1, 0, 1, 0])
def sig(x):
	return 1 / (1 + exp(-x))
for iterat in range(100000):
#after f1 and f2
	a_f1_f2 = np.dot(weights1.T, tsi[iterat % 4].T)
#after sigmoida
	a_sigm1 = sig(a_f1_f2)
#after f3
	a_f3 = np.dot(weights2.T, a_sigm1)
#after second sigmoida
	a_sigm2 = sig(a_f3)
#counting weights
	weights2 += (a_sigm2 - inp[iterat % 4]) * (a_sigm2 * (1 - a_sigm2) * weights2) * 0.001
	weights1 += np.outer(tsi[iterat % 4].T, (a_sigm2 - inp[iterat % 4]) * a_sigm2 * (1 - a_sigm2) * weights2 * \
	a_sigm1 * (1 - a_sigm1)) * 0.001
	if iterat % 10000 == 0:
		print('weights1 = ', weights1, '\n', 'weights2 = ', weights2, '\n', 'Loss = ', inp[iterat % 4] - sig(sig(weights1.T @ tsi[0].T).T @ weights2))
print('Result = ', sig(sig(weights1.T @ np.array([1, 1, 1, 1, 1, 1]).T).T @ weights2))
