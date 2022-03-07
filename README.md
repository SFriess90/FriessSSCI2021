# Predicting CMA-ES Operators as Inductive Biases for Shape Optimization Problems

Code to replicate experiments conducted for predicting CMA-ES operators in the scenario of shape optimization problems.

## Introduction

The code we provide with our repository is based upon our paper [Predicting CMA-ES Operators as Inductive Biases for Shape Optimization Problems](https://ieeexplore.ieee.org/document/9660001/) (Friess, Tiňo, Menzel, Sendhoff & Yao, 2022) and can be used to replicate the experiments for predicting CMA-ES in the scenario of simple shape optimization with mesh data. 

We provide experimental data from in the intermediate steps within our paper in the folders  `model_data`,  `ea_data` and `histogram_data` respectively. The folder `gcn` contains the necessary custom operations for graph convolution and pooling. Where the former is our own custom Keras implementation and the latter is based upon code accompanying the original paper from Defferrard et al. (2016). Different implementations of evolutionary algorithms from the DEAP library can be used for experimentation are provided in `ea_generate` folder.  The scripts `training_\*.py` can be used to generate partition models, `histogram_calculation_\*.py` to obtain structured data formats and `Notebook-\*nb` to experiment with network architectures and analyze their feature extraction capabilities. All them are contained within the main folder.

In the following, we will give a more in-depth description on the technical requirements and how to use our scripts according to the given step-by-step experiments within our paper.
 `run.py` 


## Technical Requirements

To obtain structured data formats using search space partition methods, we use a k-means implementation from the Scikit-Learn software package. For the construction of Delaunay graphs, the SciPy software package can be used. Further, optimized implementations of the self-organized map and growing neural gas are contained within the NeuPy library. For implementing the neural network architectures we use Keras with a TensorFlow backend.  

All required libraries can be installed by executing `pip install -r requirements.txt` from the main directory via the command line. 


| Library       | Description |
| ------------- |:-------------|
| TensorFlow  | Library for constructing neural network architectures. |
| Keras  | Python interface for the usage of Tensorflow.   |
| DEAP        | Provides different implementations of EAs. |
| SciPy    | Efficient calculations of distances and of Voronoi graphs.  |
| Scikit-Learn       | Implements the k-Means clustering.  |
| NeuPy       | Implements SOM and GNG. |

The following sections elaborate on how to replicate the steps and experiments presented within our paper. 

## 1. Setting up a Search Space Partition Method


## How to Cite

### Paper Reference
* Friess, S., Tiňo, P.,Menzel, S., Sendhoff, B. and Yao, X., 2022, January. Predicting CMA-ES Operators as Inductive Biases for Shape Optimization Problems. In 2021 IEEE Symposium Series on Computational Intelligence (SSCI) (pp. 1-7). IEEE.

### BibTeX Reference
```
@INPROCEEDINGS{9660001,
  author={Friess, Stephen and Tiňo, Peter and Menzel, Stefan and Sendhoff, Bernhard and Yao, Xin},
  booktitle={2021 IEEE Symposium Series on Computational Intelligence (SSCI)}, 
  title={Predicting CMA-ES Operators as Inductive Biases for Shape Optimization Problems}, 
  year={2022},
  pages={1-7}}
```

## Acknowledgements

This research has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement number 766186 (ECOLE). It was also supported by the Program for Guangdong Introducing Innovative and Enterpreneurial Teams (Grant No. 2017ZT07X386), Shenzhen Science and Technology Program (Grant No. KQTD2016112514355531), and the Program for University Key Laboratory of Guangdong Province (Grant No. 2017KSYS008).
