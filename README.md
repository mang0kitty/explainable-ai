# Explainable AI (XAI)

## Introduction

This repository contains examples and best practices for building explainable AI systems, provided as Jupyter notebooks.

In [explainable-ai](xai_algorithms), a number of utilities are included to facilitate standard tasks like importing datasets and models in the formats required by various algorithms. For self-study and customisation in your own applications, there are implementations of a number of cutting-edge algorithms available.

## Getting Started

For additional information on configuring your system locally, please refer to the [setup guide](SETUP.md).

The XAI package installation has been tested using
- Python version 3.10 and [venv](https://docs.python.org/3/library/venv.html), or [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment)

The package and its dependencies should be installed in a clean environment (such as
[conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment) or [venv](https://docs.python.org/3/library/venv.html)).

To set up on your local machine:

1. Core utilities, CPU-based algorithms, and dependencies should be installed:

1. Verify that Python libraries and the necessary applications are installed.

   + On Linux this can be supported by adding:

     ```bash
     sudo apt-get install -y build-essential libpython<version>
     ``` 

     where `<version>` should be the Python version (e.g. `3.10`).

   + On Windows you will need [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

1. Create a conda or virtual environment.  See the
   [setup guide](SETUP.md) for more details.

1. Into the created environment, install the package from
   [PyPI](https://pypi.org):

   ```bash
   pip install --upgrade pip
   pip install --upgrade setuptools
   ```

1. Register your (conda or virtual) environment with Jupyter:

   ```bash
   python -m ipykernel install --user --name my_environment_name --display-name ".venv"
   ```

1. Start the Jupyter notebook server

   ```bash
   jupyter notebook
   ```

## Algorithms

The XAI algorithms that are currently offered in the repository are listed in the table below. Under the Example column, notebooks are connected as Quick start links that display an easy-to-run example of the method or as Deep dive links that go into great detail about the math and implementation of the algorithm.


| Algorithm | Type | Description | Example | 
|-----------|------|-------------|---------|
| Gradient-weighted Class Activation Mapping (Grad-CAM) | non-model-agnostic | Algorithm that uses the gradients of any target concept (say ‘dog’ in a classification network or a sequence of words in captioning network) flowing into the final convolutional layer, to produce a coarse localization map highlighting the important regions in the image for predicting the concept. It works only with CNNs. | [Quick start](examples/00_quick_start/grad_cam_torch.ipynb) / [Deep dive](examples/01_deep_dive/grad_cam_torch.ipynb) |
| Local Interpretable Model-agnostic (LIME) | model-agnostic | LIME  is a visualization technique that helps explain individual predictions. It is model agnostic so it can be applied to any supervised regression or classification model. | [Quick start](examples/00_quick_start/lime.ipynb) | 
| Layer-wise Relevance Propagation (LRP) | non-model-agnostic | LRP is a technique that brings such explainability and scales to potentially highly complex deep neural networks. It operates by propagating the prediction backward in the neural network, using a set of purposely designed propagation rules | [Quick start](examples/00_quick_start/lrp.ipynbb) | 
| Anchors | model-agnostic | Anchors are high precision explainers that use reinforcement learning methods to come up with the set of feature conditions (called anchors), which will help explain the observation of interest and also a set of surrounding observations with a high precision (the user is free to choose their minimum precision cut-off). | [Quick start](examples/00_quick_start/anchors.ipynb) | 


### Algorithm Comparison

## Contributing

We welcome contributions and ideas for this project. Please review our [contribution guidelines](CONTRIBUTING.md) before contributing.

## Related projects

## Reference papers
- [Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers](https://arxiv.org/abs/1604.00825)
- [Anchors](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)
