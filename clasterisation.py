import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist
import sklearn

data = load_iris()


#k-means с использованием евклидовой метрикой
def kmeans(metric='euclidean', show=False):
	np.random.seed(seed=42)
	centroids = np.random.normal(loc=3.0, scale=1., size=12)
	centroids = centroids.reshape((3, 4))
	cent_history = []
	cent_history.append(centroids)

	distances = cdist(data.data, centroids, metric=metric)
	labels = distances.argmin(axis=0)

	iter = 0
	centroids2 = np.zeros((3, 4))
	while not np.array_equal(centroids, centroids2):
		centroids2 = centroids.copy()
		# Считаем расстояния от наблюдений до центроид
		distances = cdist(data.data, centroids, metric=metric)
		# Смотрим, до какой центроиды каждой точке ближе всего
		labels = distances.argmin(axis=1)
		# Положим в каждую новую центроиду геометрический центр её точек
		centroids = centroids.copy()
		centroids[0, :] = np.mean(data.data[labels == 0], axis=0)
		centroids[1, :] = np.mean(data.data[labels == 1], axis=0)
		centroids[2, :] = np.mean(data.data[labels == 2], axis=0)
	
		cent_history.append(centroids)
		iter += 1
	if show:
		fig = plt.figure(figsize=(8, 8))
		fig.suptitle('k-means method, ' + metric + ' metric', fontsize=12)
		for i in range(iter):
			distances = cdist(data.data, centroids, metric=metric)
			labels = distances.argmin(axis=1)
			if i <= 3:
				ax = fig.add_subplot(220+i+1, projection='3d')
				ax.scatter(data.data[labels == 0, 0], data.data[labels == 0, 1], data.data[labels == 0, 2], 'bo', label='cluster #1')
				ax.scatter(data.data[labels == 1, 0], data.data[labels == 1, 1], data.data[labels == 1, 2], 'co', label='cluster #2')
				ax.scatter(data.data[labels == 2, 0], data.data[labels == 2, 1], data.data[labels == 2, 2], 'mo', label='cluster #3')
				ax.scatter(cent_history[i][:, 0], cent_history[i][:, 1], cent_history[i][:, 2], 'rX')
				ax.legend(loc=0)
				plt.title(f'Step {i+1}')
		plt.show()
		#последняя итерация алгоритма
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(projection='3d')
		ax.scatter(data.data[labels == 0, 0], data.data[labels == 0, 1], data.data[labels == 0, 2], 'bo', label='cluster #1')
		ax.scatter(data.data[labels == 1, 0], data.data[labels == 1, 1], data.data[labels == 1, 2], 'co', label='cluster #2')
		ax.scatter(data.data[labels == 2, 0], data.data[labels == 2, 1], data.data[labels == 2, 2], 'mo', label='cluster #3')
		ax.scatter(cent_history[iter-1][:, 0], cent_history[iter-1][:, 1], cent_history[iter-1][:, 2], 'rX')
		plt.title('Final step in kmeans, ' + metric + ' metric')
		plt.legend(loc=0)
		plt.show()
	return labels

#DBSCAN с евклидовой метрикой
from sklearn.cluster import DBSCAN

def dbscan_meth(metric='euclidean', show=False):
	clustering = DBSCAN(eps=0.5, min_samples=5, metric=metric).fit(data.data)
	if show:
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(projection='3d')
		ax.scatter(data.data[clustering.labels_ == -1, 0], data.data[clustering.labels_ == -1, 1], data.data[clustering.labels_ == -1, 2], 'bo', label='cluster #1')
		ax.scatter(data.data[clustering.labels_ == 0, 0], data.data[clustering.labels_ == 0, 1], data.data[clustering.labels_ == 0, 2], 'co', label='cluster #2')
		ax.scatter(data.data[clustering.labels_ == 1, 0], data.data[clustering.labels_ == 1, 1], data.data[clustering.labels_ == 1, 2], 'mo', label='cluster #3')
		ax.scatter(data.data[clustering.labels_ == 2, 0], data.data[clustering.labels_ == 2, 1], data.data[clustering.labels_ == 2, 2], 'ko', label='cluster #4')
		plt.title(f'DBSCAN method, ' +  metric + ' metric')
		plt.legend(loc=0)
		plt.show()
	return clustering.labels_

#сравниваем внешние оценки
def external_eval(label):
	return sklearn.metrics.adjusted_mutual_info_score(data.target, label)


#сравниваем внутренние оценки, попарное евклидово расстояние
def internal_eval(labels):
	sum_norm = 0
	diff = 0
	for item in np.unique(labels):
		main = data.data[labels == item]
		diff += cdist(main, data.data[labels != item]).sum()
		for i in range(main.shape[0] - 1):
			for j in range(i + 1, main.shape[0]):
				sum_norm += np.linalg.norm(main[i] - main[j])
	
	return sum_norm, diff

print('Внешняя оценка kmeans, euclidean - ', external_eval(kmeans()))
print('Внешняя оценка kmeans, cityblock - ', external_eval(kmeans('cityblock')))
print('Внешняя оценка DBSCAN, euclidean - ', external_eval(dbscan_meth('euclidean', False)))
print('Внешняя оценка DBSCAN, cityblock - ', external_eval(dbscan_meth('cityblock', False)))
print('Внутренняя оценка kmeans, euclidean - ', internal_eval(kmeans()))
print('Внутренняя оценка kmeans, cityblock - ', internal_eval(kmeans('cityblock')))
print('Внутренняя оценка DBSCAN, euclidean - ', internal_eval(dbscan_meth('euclidean', False)))
print('Внутренняя оценка DBSCAN, cityblock - ', internal_eval(dbscan_meth('cityblock', False)))

#графическое сравнение
def graph_comp(lab_ke, lab_kc, lab_de, lab_dc):
	fig = plt.figure(figsize=(8, 8))
	fig.suptitle('Сравнение методов', fontsize=12)
	ax = fig.add_subplot(221, projection='3d')
	ax.scatter(data.data[lab_ke == 0, 0], data.data[lab_ke == 0, 1], data.data[lab_ke == 0, 2], 'bo', label='cluster #1')
	ax.scatter(data.data[lab_ke == 1, 0], data.data[lab_ke == 1, 1], data.data[lab_ke == 1, 2], 'co', label='cluster #2')
	ax.scatter(data.data[lab_ke == 2, 0], data.data[lab_ke == 2, 1], data.data[lab_ke == 2, 2], 'mo', label='cluster #3')
	ax.legend(loc=0)
	plt.title('kmeans method, euclidean metric')
	ax = fig.add_subplot(222, projection='3d')
	ax.scatter(data.data[lab_kc == 0, 0], data.data[lab_kc == 0, 1], data.data[lab_kc == 0, 2], 'bo', label='cluster #1')
	ax.scatter(data.data[lab_kc == 1, 0], data.data[lab_kc == 1, 1], data.data[lab_kc == 1, 2], 'co', label='cluster #2')
	ax.scatter(data.data[lab_kc == 2, 0], data.data[lab_kc == 2, 1], data.data[lab_kc == 2, 2], 'mo', label='cluster #3')
	ax.legend(loc=0)
	plt.title('kmeans method, cityblock metric')
	ax = fig.add_subplot(223, projection='3d')
	ax.scatter(data.data[lab_de == 0, 0], data.data[lab_de == 0, 1], data.data[lab_de == 0, 2], 'bo', label='cluster #1')
	ax.scatter(data.data[lab_de == 1, 0], data.data[lab_de == 1, 1], data.data[lab_de == 1, 2], 'co', label='cluster #2')
	ax.scatter(data.data[lab_de == 2, 0], data.data[lab_de == 2, 1], data.data[lab_de == 2, 2], 'mo', label='cluster #3')
	ax.legend(loc=0)
	plt.title('DBSCAN method, euclidean metric')
	ax = fig.add_subplot(224, projection='3d')
	ax.scatter(data.data[lab_dc == 0, 0], data.data[lab_dc == 0, 1], data.data[lab_dc == 0, 2], 'bo', label='cluster #1')
	ax.scatter(data.data[lab_dc == 1, 0], data.data[lab_dc == 1, 1], data.data[lab_dc == 1, 2], 'co', label='cluster #2')
	ax.scatter(data.data[lab_dc == 2, 0], data.data[lab_dc == 2, 1], data.data[lab_dc == 2, 2], 'mo', label='cluster #3')
	ax.legend(loc=0)
	plt.title('DBSCAN method, cityblock metric')
	plt.show()
graph_comp(kmeans('euclidean', True), kmeans('cityblock', True), \
			dbscan_meth('euclidean', True), dbscan_meth('cityblock', True))
