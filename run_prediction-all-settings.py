import numpy as np
import meshdeform as md
import optimization as opt
import time
import pickle
import multiprocessing
import concurrent.futures
import copy
import configurator as conf

for s in ['2']:
#for s in ['2']:

    input_shapes = pickle.load(open('shapes'+str(s)+'.out', 'rb'))
    input_base = pickle.load(open('base'+str(s)+'.out', 'rb'))


    def deformMidsection3D(cp_in, input_x):
        xs, ys, zs = np.shape(cp_in)[0], np.shape(cp_in)[1], np.shape(cp_in)[2]

        delta = np.zeros(shape=(2, 1, 2, 3))

        delta[0][0][0] = np.array([-input_x[0], 0, 0])
        delta[1][0][0] = np.array([input_x[0], 0, 0])
        delta[0][0][1] = np.array([-input_x[1], 0, input_x[2]])
        delta[1][0][1] = np.array([input_x[1], 0, input_x[2]])

        y0 = 3
        yE = ys - 2

        for x in range(0, xs):
            for y in range(y0, yE):
                for z in range(0, zs):
                    cp_in[x][y][z] = np.array([cp_in[x][y][z][0], cp_in[x][y][z][1], cp_in[x][y][z][2]]) + delta[x][0][z]

    experiments = []

    objective = md.buildObjectivefunction(input_shapes, input_base, deformMidsection3D)

    def runExperiment():
        a = time.time()
        popmins, pops, covs, sigmas, centroids = opt.runCMAES(objective, [0] * 3, config=[10, 10, 1.0], gen=10)
        #print(centroids)

        b = time.time()
        print((b - a) / 60)

        pred_sigma, pred_cov = conf.predictConfiguration(pops)
        popmins2, pops2, _, _, _ = opt.runCMAES(objective, [0] * 3, config=[10, 10, pred_sigma], gen=10)
        popmins3, pops3, _, _, _ = opt.runCMAES(objective, [0] * 3, config=[10, 10, 1.0], gen=10, cmat=pred_cov)
        popmins4, pops4, _, _, _ = opt.runCMAES(objective, [0] * 3, config=[10, 10, pred_sigma], gen=10, cmat=pred_cov)
        c = time.time()
        print((c - b) / 60)

        print("Finished experiment.")
        
        return str([popmins, popmins[:0] + popmins2[:], popmins[:0] + popmins3[:], popmins[:0] + popmins4[:]])

    if __name__ == '__main__':
        print("Main function has been called.")

        experimental_data = []

        start = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(runExperiment) for _ in range(0,10)]

            for f in concurrent.futures.as_completed(results):
                result = f.result()
                experimental_data.append(eval(result))

        end = time.time()
        print(f'Main function finished after: {(end-start)/60} minutes')

        out_experiments = open('experimental_data/experimental_data-predict-10-'+str(s)+'-all-10-offset-0.out', 'wb')
        pickle.dump(experimental_data, out_experiments)
        out_experiments.close()