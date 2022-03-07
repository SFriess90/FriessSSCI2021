import array
import random

import numpy as np
import matplotlib.pyplot as plt

from deap import base, benchmarks
from deap import creator
from deap import tools
#from deap import cma

import customalgorithm as algorithms
import customcma as cma


def runCMAES(objective, init, gen=100, config=[10, 10, 1.0], cmat=np.full((2,2),None)):

    in_mu, in_lambda, in_sigma= config
    
    if (cmat==None).any():
        cmat=np.identity(len(init))
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", objective)

    strategy = cma.Strategy(centroid=init, sigma=in_sigma, lambda_=in_lambda,mu=in_mu, cmatrix=cmat)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda ind: [ind,ind.fitness.values[0]])
    
    def returnBestIndividual(inds):     
        idx = np.argmin([x[1] for x in inds])
        return inds[idx]
    
    def returnPopulation(inds):     
        return [[x[0],x[1]] for x in inds]
    
    stats.register("returnBest", returnBestIndividual)
    stats.register("returnPop", returnPopulation) 

    upd = algorithms.eaGenerateUpdate(toolbox, ngen=gen, stats=stats, verbose=False)
    popfit = [upd[1][i]['returnBest'] for i in range(len(upd[1]))]
    pops = [upd[1][i]['returnPop'] for i in range(len(upd[1]))]
    
    return popfit, pops, upd[-3], upd[-2], upd[-1]