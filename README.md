# Analysis Manager

A small python based tool for managing scientific and engineering analysis.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Background

This was a core capability of the Hybrid Advanced Reactor Evaluator ([HARE](https://github.com/tokasamwin/hare)) tool and has been separated to provide a more general capability.

## Install
TODO: publish on PyPI

```
pip install pip@git+https://github.com/tokasamwin/analysis-manager
```



## Usage


The aim of this is to provide a simple interface for turning basic scripts into parametric workflows which intuitively match how engineering and scientific analysis is carried out:

* In stages, with only successful design points proceeding to later stages
* Within a design space represented by parameters and their bounds
* Used for parameter searches, design point optimisation, or surrogate modelling

These aspects are all implemented in a modular way. Get your workflow working with a search over a broad parameter space, narrow your parameter bounds and carry out a Multi-Objective Optimisation to find a Pareto front, or update the workflow to use a more detailed assessment stage. 

### Core concepts

* **Sampler**: defines the parameters to be sampled, including continuous and discrete/categorical parameters. This provides an interface to generate a DataFrame of inputs sampled in different methods (currently only using Grid and Sobol sampling methods)
* **Stages**: scientific and engineering analysis is typically staged, including pre-processing, processing and post-processing stages. Wrap your functions in a stage and optionally provide a gate:
    * These gates could be parameter gates, or a check for whether a processing stage successfully completed, or other types of validation
    * Chain stages together, such as building CAD, conducting a transient thermal assessment, a structural assessment, and finally a creep-fatigue assessment for data points which have acceptable temperatures and maximum stresses
* **Routines**: search across your parameter space, optimise across it, or generate a surrogate model. Optimisation uses wrapped methods from [PyMOO](https://pymoo.org)
* **Run manager**: a central run manager keeps track of your design points and provides the relevant path to store data for each assessment:
    * The run manager can chain different routines, such as a search routine leading to a starting point for an optimisation routine

### Run Manager structure

```
|- Design Point
   |- Stage
      |- Assessment files
      |- Input.json
   |- Run configuration data
```

## Contributing

PRs accepted.

## License

[MIT © Sam Ha](../LICENSE)