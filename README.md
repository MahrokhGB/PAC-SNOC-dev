# PAC-SNOC
PyTorch implementation of our PAC-Bayesian framework for optimal control with stability guarantees.

## Implementation details

????? The [implementation details](docs/implementation_details.pdf) can be found in the ```docs``` folder.

## Installation

```bash
git clone https://github.com/DecodEPFL/PAC-SNOC.git

cd PAC-SNOC

python setup.py install
```

## Basic usage

Two environments are proposed to train the controllers with our approach.
The two systems are:
\textit{i)} a simple Linear Time-Invariant (LTI) system, and
\textit{ii)} a system of two planar robots navigating to prespecified locations while avoiding collisions.

### LTI system

To train the controllers, run the following script:
```bash
./run.py --sys_model [SYS_MODEL]
```
where available values for `SYS_MODEL` are `corridor` and `robots`.

### Robots

1. Generate the data by running
```bash
python experiments/robotsX/generate_data.py
```
2. Train the empirical controller by running
```bash
python experiments/robotsX/run_emp.py
```
3. Train the controllers using our apporach by running
```bash
python experiments/robotsX/run_SVGD.py
```

## Examples:

### LTI system

### Robots

The following gifs show trajectories of the 2 robots before (left) and after the training of the empirical controller (middle),
and using our approach (right).
The agents that need to coordinate in order to pass through a narrow passage.
The initial conditions used as training data are marked with &#9675;.
They consist of `s = 30` samples starting from a random initial position marked with &#9675;, sampled from a Normal distribution centered in
[&#177;2 , -2] with standard deviation of 0.2.
The shown trajectories start from a random initial position sampled from the test data and marked with
&#11043;.




<p align="center">
<img src="./experiments/robotsX/gif/ol.gif" alt="robot_trajectories_before_training" width="255"/>
<img src="./experiments/robotsX/gif/emp.gif" alt="robot_trajectories_using_empirical_controller" width="255"/>
<img src="./experiments/robotsX/gif/svgd.gif" alt="robot_trajectories_using_our_controller" width="255"/>
</p>

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
