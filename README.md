# Predicting CMA-ES Operators as Inductive Biases for Shape Optimization Problems

Code to replicate experiments conducted for predicting CMA-ES operators in the scenario of shape optimization problems.

## Description

The code we provide within our repository is based upon our paper [Predicting CMA-ES Operators as Inductive Biases for Shape Optimization Problems](https://ieeexplore.ieee.org/document/9660001/) (Friess, Tiňo, Menzel, Sendhoff & Yao, 2022) and can be used to replicate the experiments for predicting CMA-ES in the scenario of simple shape optimization with mesh data. 

The notebook `21-07-19-CMAES.ipynb` allows the replication of the experiments to derive an improved operator configuration for the CMA-ES. To conduct some simple shape optimization experiments the notebook `21-07-10-mesh-deforming.ipynb` can be used, in conjunction with test shapes provided in `meshes` folder.  

The operator prediction experiments self can be replicated using the script `run_prediction-all-settings.py` from the command line. Upon running, it a shape optimization experiment is started using either the shape with increased volume either by heighting (2) or widening the base shape (3) as target. Particularly, the baseline is computed first, as we all as a second experimental run is conducted using predicted operator configurations (step-size and covariance matrix) after a pre-defined number of steps is conducted. To predict an operator configuration, the 'predictConfiguration' method from the script `configurator.py` is called with procedural optimization data as input. Upon running, first a search space partition is loaded to from the folder `model_data` to convert the data into structured format. Subsequently, a pre-trained prediction model  `prediction_model.out` from the root folder is loaded,  from which using the operator configuration is predicted using the structured input data.

Prediction models can be generated using the notebook `ConvertData-Regression2.ipynb`. For the training a prediction model, the script `histogram_calculation_gng.py` can be used to explicitly convert any experimental data in the folder `experimental_data` into structured data formats in the folder  `histogram_data`. 

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
