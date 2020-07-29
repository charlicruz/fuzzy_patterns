import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn import datasets
from genetic_algorithm import genetic_algorithm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import csv

import pandas as pd

# https://github.com/nathanrooy/particle-swarm-optimization
import pso_simple

def normalize_dataset(dataset):
    min_arr = np.amin(dataset, axis=0)
    return (dataset - min_arr) / (np.amax(dataset, axis=0) - min_arr)


# def evaluate_new_fuzzy_system(w1, w2, w3, w4, data, target):

# 	universe = np.linspace(0, 1, 100)

# 	x = []
# 	for w in [w1, w2, w3, w4]:
# 		x.append({'s': fuzz.trimf(universe, [0.0, 0.0, w]),
# 		          'm': fuzz.trimf(universe, [0.0, w, 1.0]),
# 			      'l': fuzz.trimf(universe, [w, 1.0, 1.0])})

# 	x_memb = []
# 	for i in range(4):
# 		x_memb.append({})
# 		for t in ['s', 'm', 'l']:
# 			x_memb[i][t] = fuzz.interp_membership(universe, x[i][t], data[:, i])

# 	is_setosa = np.fmin(np.fmax(x_memb[2]['s'], x_memb[2]['m']), x_memb[3]['s'])
# 	is_versicolor = np.fmax(np.fmin(np.fmin(np.fmin(np.fmax(x_memb[0]['s'], x_memb[0]['l']), np.fmax(x_memb[1]['m'], x_memb[1]['l'])), np.fmax(x_memb[2]['m'], x_memb[2]['l'])),x_memb[3]['m']), np.fmin(x_memb[0]['m'], np.fmin(np.fmin(np.fmax(x_memb[1]['s'], x_memb[1]['m']),x_memb[2]['s']), x_memb[3]['l'])))
# 	is_virginica = np.fmin(np.fmin(np.fmax(x_memb[1]['s'], x_memb[1]['m']), x_memb[2]['l']), x_memb[3]['l'])

# 	result = np.argmax([is_setosa, is_versicolor, is_virginica], axis=0)

# 	return (result == target).mean()

def evaluate_new_fuzzy_system(ws, data, target):
    universe = np.linspace(0, 1, 100)
    x = []
    for w in ws:
		x.append({'s': fuzz.trimf(universe, [0.0, 0.0, w]),
		          'm': fuzz.trimf(universe, [0.0, w, 1.0]),
			      'l': fuzz.trimf(universe, [w, 1.0, 1.0])})

    # membership
    x_memb = []
    for i in range(len(ws)):
        x_memb.append({})
        for t in ['s', 'm', 'l']:
            x_memb[i][t] = fuzz.interp_membership(universe, x[i][t], data[:, i])

    # you need to create your rules here

    # you need to evaluate the result here
    #  result = np.argmax([is_something, is_another_thing, is_another_thing_2], axis=0)

    # finally, evaluate the system
    # return (result == target).mean()


def main():
    # iris = datasets.load_iris()


    # iris = np.array([[0.99394,1,0.77212,0.77455],
    #               [0.99394,1,0.78182,0.79394],
    #               [0.35758,0.36364,0.26061,0.25455],
    #               [0.21212,0.2484,0.27879,0.24242],
    #               [0.125450,0.12182,0.006060,0.01818],
    #               [0.13333,0.12727,0.03636,0]])
	#iris = datasets.load_iris()

    dataset = pd.read_csv('random_example2.csv', sep=",")

    # iris = load_my_dataset()

	#iris = datasets.load_iris()

    normalized_dataset = normalize_dataset(dataset)

    n_features = normalized_dataset.shape[1]

    fitness = lambda w: 1.0 - evaluate_new_fuzzy_system(w, dataset, dataset.target)

	# Test Fuzzy
    #w = [0.07, 0.34, 0.48, 0.26] # 95%
    #w = [0, 0.21664307088134033, 0.445098590128248, 0.2350617110613577] # 96.6%
    #print(1.0 - fitness(w))

    record = {'GA': [], 'PSO': []}

    for _ in tqdm(range(30)):

		# GA
        t, fbest = genetic_algorithm(fitness_func=fitness, dim=n_features, n_individuals=10, epochs=30, verbose=False)
        record['GA'].append(1.0 - fbest)

		# PSO
        initial=[0.5, 0.5, 0.5, 0.5]
        bounds=[(0, 1), (0, 1), (0, 1), (0, 1)]
        best, fbest = pso_simple.minimize(fitness, initial, bounds, num_particles=10, maxiter=30, verbose=False)
        record['PSO'].append(1.0 - fbest)


	# Statistcs about the runs
    print('GA:')
    print(np.amax(record['GA']), np.amin(record['GA']))
    print(np.mean(record['GA']), np.std(record['GA']))

    print('PSO:')
    print(np.amax(record['PSO']), np.amin(record['PSO']))
    print(np.mean(record['PSO']), np.std(record['PSO']))


    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(list(record.values()), vert=True, patch_artist=True, labels=list(record.keys()))

    ax.set_xlabel('Algoritmo')
    ax.set_ylabel('Acurácia')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
	main()