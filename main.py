#carlos cruz / 01082020
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
import pso_simple

def normalize_dataset(dataset):
    min_arr = np.amin(dataset, axis=0)
    return (dataset - min_arr) / (np.amax(dataset, axis=0) - min_arr)

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

    # MY RULES ###########
    # R2 = x7 = x8 = long and x15 = x16 = middle and x23 = x24 = short then Efficient
    # What I understood:
    #
    #
    # x7 is long AND x8 is LONG AND x15 is middle AND x16 is middle AND x23 is midle AND x24 is short -> efficient
    #
    #
    # Since logical OR become a MAX and logical AND becomes MIN
    # then we should have something like:
    is_efficient =  np.fmin(
        x_memb[21]['l'],
        np.fmin(
            x_memb[19]['l'],
            np.fmin(
                x_memb[20]['m'],
                np.fmin(
                    x_memb[18]['m'],
                    np.fmin(
                        x_memb[22]['m'],
                        x_memb[23]['s']
                    )
                )
            )
        )
    )

    # R1 = x1 = x2 = long and X5 = x6 = long and x9 = x10 = middle and x17 = x18 = short then Inefficient
    is_inefficient = np.fmin(
        x_memb[0]['l'],
        np.fmin(
            x_memb[1]['l'],
            np.fmin(
                x_memb[4]['l'],
                np.fmin(
                    x_memb[5]['l'],
                    np.fmin(
                        x_memb[2]['m'],
                        np.fmin(
                            x_memb[7]['m'],
                            np.fmin(
                                x_memb[3]['s'],
                                x_memb[6]['s'],
                            )
                        )
                    )
                )
            )
        )
    )

    # R3 = x3 = x4 = long and x11 = x12 = middle and x13 = x14 = middle and x19 = x20 = short and x21 = x22 = short then Mixt
    is_mixed = np.fmin(
        x_memb[14]['l'],
        np.fmin(
            x_memb[15]['l'],
            np.fmin(
                x_memb[10]['m'],
                np.fmin(
                    x_memb[11]['m'],
                    np.fmin(
                        x_memb[12]['m'],
                        np.fmin(
                            x_memb[13]['m'],
                            np.fmin(
                                x_memb[18]['s'],
                                np.fmin(
                                    x_memb[19]['s'],
                                    np.fmin(
                                        x_memb[9]['l'],
                                        x_memb[8]['l'],
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    result = np.argmax([is_efficient, is_mixed, is_inefficient], axis=0)
    return (result == target).mean()

def load_dataset(filename):
    raw_dataset = pd.read_csv(filename)
    data = normalize_dataset(raw_dataset.values[:, :-1])
    target = raw_dataset['TARGET'].values
    return data, target

def main():

    data, target = load_dataset('t.csv')#random_example_test
    normalized = normalize_dataset(data)
    n_features = data.shape[1]

    fitness = lambda ws: 1.0 - evaluate_new_fuzzy_system(ws, data, target)
	# Test Fuzzy
    #ws = [0.07, 0.34, 0.48, 0.26,0.07, 0.34, 0.48, 0.26,0.07, 0.34, 0.48, 0.26,0.07, 0.34, 0.48, 0.26,0.07, 0.34, 0.48, 0.26,0.07, 0.34, 0.48, 0.26] # 95%

    ws = [0.993900000000000,1,0.772120000000000,0.774550000000000,0.993940000000000,1,0.781820000000000,0.793949000000000,0.357580000000000,0.363640000000000,0.260610000000000,0.254550000000000,0.212120000000000,0.248480000000000,0.278790000000000,0.242420000000000,0.125450000000000,0.121820000000000,0.00606000000000000,0.0181800000000000,0.133330000000000,0.127270000000000,0,0]    # w = [0, 0.21664307088134033, 0.445098590128248, 0.2350617110613577] # 96.6%
    Classification=1.0 - fitness(ws)
    print(Classification)
    

    record = {'GA': [], 'PSO': []}
    for _ in tqdm(range(10)):
		# GA
        t, fbest = genetic_algorithm(fitness_func=fitness, dim=n_features, n_individuals=10, epochs=40, verbose=False)
        record['GA'].append(1.0 - fbest)
		# PSO
        initial=[0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5]
        bounds=[(0, 1), (0, 1), (0, 1), (0, 1),(0, 1), (0, 1), (0, 1), (0, 1),(0, 1), (0, 1), (0, 1), (0, 1),(0, 1), (0, 1), (0, 1), (0, 1),(0, 1), (0, 1), (0, 1), (0, 1),(0, 1), (0, 1), (0, 1), (0, 1)]
        best, fbest = pso_simple.minimize(fitness, initial, bounds, num_particles=10, maxiter=30, verbose=False)
        record['PSO'].append(1.0 - fbest)
#     print(t)
    plt.plot(t)
    #print(fbest)
 	# Statistcs about the runs
    print('GA:')
    print(np.amax(record['GA']), np.amin(record['GA']))
    print(np.mean(record['GA']), np.std(record['GA']))

    #record['Classification']
    print('PSO:')
    print(np.amax(record['PSO']), np.amin(record['PSO']))
    print(np.mean(record['PSO']), np.std(record['PSO']))
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(list(record.values()), vert=True, patch_artist=True, labels=list(record.keys()))

    ax.set_xlabel('Comparison')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

# import numpy as np
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl

# # New Antecedent/Consequent objects hold universe variables and membership
# # functions
# quality = ctrl.Antecedent(np.arange(0, 11, 1), 'pattern')
# service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
# tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# # Auto-membership function population is possible with .automf(3, 5, or 7)
# quality.automf(3)
# service.automf(3)

# # Custom membership functions can be built interactively with a familiar,
# # Pythonic API
# tip['inneficient'] = fuzz.trimf(tip.universe, [0, 0, 13])
# tip['mixed'] = fuzz.trimf(tip.universe, [0, 13, 25])
# tip['efficient'] = fuzz.trimf(tip.universe, [13, 25, 25])
# # You can see how these look with .view()
# quality['average'].view()

if __name__ == '__main__':
	main()