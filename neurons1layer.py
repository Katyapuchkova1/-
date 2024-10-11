from numpy import exp, array, random, dot
import numpy as np
#сеть с двумя нейронами
tsi = np.array([  [1, 1, 1, 1, 1, 0] ])
weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
inp = np.array([1])
def sig(x):
	return 1 / (1 + exp(-x))
iterat = 0
Loss = 1
while Loss > 0.00001:
#after f1
	a_f1 = np.dot(tsi[0], weights.T)
#after sigmoida
	a_sigm = sig(a_f1)
#counting weight 
	weights += -(a_sigm - inp[0]) * a_sigm * (1 - a_sigm) * tsi[0] * 0.001
	if iterat % 1000==0:
		print('weights = ', weights, '\nLoss = ', (inp[0] - sig(tsi[0] @ weights.T))**2/2, '\n')
	iterat += 1
	Loss = (inp[0] - sig(tsi[0] @ weights.T))**2/2
print('Result = ', sig(tsi[0] @ weights.T), '\n', \
	  'Loss = ', Loss, '\n', \
	  'Iteration = ', iterat)
