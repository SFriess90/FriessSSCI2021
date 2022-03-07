from clustering_som import somPartition
from clustering_gng import gngPartition

import numpy as np
import pickle

gng = pickle.load(open("model_data/100-3-model_gng.out", "rb"))

method = "gng"
benchmarks = ["","2"]

for benchmark in benchmarks:

    file_path_pops = "experimental_data/experiment-populations" + benchmark + ".out"
    file_path_fits = "experimental_data/experiment-fitnesses" + benchmark + ".out"

    pops = pickle.load(open(file_path_pops, "rb"))
    fits = pickle.load(open(file_path_fits, "rb"))

    delta_histograms = []

    for pop, fit in zip(pops, fits):
        matrices = [gng.calculateNeuralGasHistogram(list(p), list(f)) for p, f in zip(pop, fit)]
        delta_matrices = np.array(matrices[1]) - np.array(matrices[0])
        delta_histograms.append(delta_matrices)

        # Write histogram into external file
    out_path_h = "histogram_data/histogram-experiment-populations" + benchmark + ".out"

    with open(out_path_h, 'wb') as fp:
        pickle.dump(delta_histograms, fp)

    print("Done with " + str(benchmark) + str("!"))