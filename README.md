# PAC-SNOC
PyTorch implementation of our PAC-Bayesian framework for optimal control with stability guarantees.


## Installation

```bash
git clone https://github.com/DecodEPFL/PAC-SNOC.git

cd PAC-SNOC

python setup.py install
```

## Examples:

### Robots

The following gifs show trajectories of the 2 robots before training (left), and after the training of the empirical controller (middle),
and using our approach (right).
The agents that need to coordinate in order to pass through a narrow passage while avoiding collisions between them.
The initial conditions used as training data are marked with &#9675;, 
consisting of `s = 30` samples.
The shown trajectories start from a random initial position sampled from the test data and marked with
&#11043;.


<p align="center">
<img src="./experiments/robotsX/gif/ol.gif" alt="robot_trajectories_before_training" width="255"/>
<img src="./experiments/robotsX/gif/emp.gif" alt="robot_trajectories_using_empirical_controller" width="255"/>
<img src="./experiments/robotsX/gif/svgd.gif" alt="robot_trajectories_using_our_controller" width="255"/>
</p>

## Basic usage

Two environments are proposed to train the controllers with our approach.
The two systems are:
_i_) a simple Linear Time-Invariant (LTI) system, and
_ii_) a system of two planar robots navigating to prespecified locations while avoiding collisions.

### LTI system

1. Generate the data by running
```bash
python experiments/scalar/generate_data.py
```
2. Train the empirical controller by running
```bash
python experiments/scalar/run_empirical.py
```
3. Train the benchmark controller by running
```bash
python experiments/scalar/run_benchmark.py
```
4. Train the controllers using our appoach by running
```bash
python experiments/scalar/run_grid_Gibbs.py
```

### Robots

1. Generate the data by running
```bash
python experiments/robotsX/generate_data.py
```
2. Train the empirical controller by running
```bash
python experiments/robotsX/run_emp.py
```
3. Train the controllers using our appoach by running
```bash
python experiments/robotsX/run_SVGD.py
```

## Pre-trained models

Pre-trained models of the LTI system and robots can be found in the folders 
[robotsX](experiments/robotsX/saved_results/trained_models) and 
[scalar](experiments/scalar/saved_results), respectively.  


## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


## References
[[1]](https://arxiv.org/pdf/????.?????.pdf) Mahrokh Ghoddousi Boroujeni, Clara Galimberti, Andreas Krause, Giancarlo
Ferrari-Trecate. "A PAC-Bayesian Framework for Optimal Control with Stability Guarantees," 2024.
