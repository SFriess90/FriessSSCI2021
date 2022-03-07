import pickle
import numpy as np

def constructCovariance(inputs):

    ind, ino = inputs[:3], inputs[3:]  
    diag = np.array([[ind[0],0,0],[0,ind[1],0],[0,0,ind[2]]])
    offdiag = np.array([[0,ino[0],ino[2]],[0,0,ino[1]],[0,0,0]])
    
    return offdiag+offdiag.T+diag

def predictConfiguration(pops):
    fac = 12
    shift = np.array([2.5, 2.5, 2.5])

    a, b, c = 0, 3, 2

    poplist = fac * np.array([[ind[0] for ind in gen] for gen in pops[a:b:c]] - shift)
    fitlist = np.array([[ind[1] for ind in gen] for gen in pops[a:b:c]])

    gng = pickle.load(open("model_data/100-3-model_gng.out", "rb"))

    matrices = [gng.calculateNeuralGasHistogram(list(p), list(f)) for p, f in zip(poplist, fitlist)]
    delta_matrix = (np.array(matrices[1]) - np.array(matrices[0]))

    out_path_h = "prediction-model.out"
    model = pickle.load(open(out_path_h, "rb"))
    config = model.predict(delta_matrix[:1])

    sigma, cov = config[1][0][0], constructCovariance(config[0][0])

    return sigma, cov