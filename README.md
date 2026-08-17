# Comparison of Explainability Methods for Classification Tasks

This repository contains the Python implementation developed for a Master's thesis comparing different Explainable Artificial Intelligence (XAI) methods for machine learning classification tasks.

## Overview

The study compares three post-hoc explainability methods:

* LIME (Local Interpretable Model-Agnostic Explanations)
* SHAP (SHapley Additive exPlanations)
* Counterfactual Explanations

The methods are evaluated using controlled synthetic classification datasets and a real-world fringe classification use case based on Freeman Chain Code (FCC) features.

## Classification Models

The experiments use the following classification models:

* Random Forest
* Naive Bayes

For the synthetic experiments, Gaussian Naive Bayes is used. For the FCC-based classification, Categorical Naive Bayes and Random Forest are used.

## Evaluation Criteria

The explainability methods are compared using quantitative criteria including:

* Fidelity
* Stability
* Sparsity

For counterfactual explanations, validity is also evaluated to determine whether the generated counterfactual successfully changes the model prediction to the desired class.

## Repository Structure

### `Synthetic_Experiments/`

Contains the implementation for the synthetic classification experiments.

Three dataset scenarios are considered:

* Clean dataset
* Tight-boundary dataset
* Combined dataset containing clean data, tight-boundary samples, and outliers

The folder contains scripts for dataset generation, classification using Random Forest and Gaussian Naive Bayes, and explanation using LIME, SHAP, and Counterfactual Explanations.

### `FCC_Experiments/`

Contains the implementation for the FCC-based fringe classification use case.

The folder includes scripts for:

* Freeman Chain Code processing and feature extraction
* Fringe labeling
* Classification
* LIME explanations
* SHAP explanations
* Counterfactual explanations

## FCC Dataset

The FCC experiments use an interferogram stored in the original `2K_Horizontal.mat` data file.

The original interferogram data file is **not included in this repository**. The required data must therefore be provided separately before running the FCC preprocessing pipeline.

The FCC implementation expects the MATLAB file to be available as:

`2K_Horizontal.mat`

The MATLAB variable used for the interferogram is:

`I_x`

## Requirements

The implementation was developed in Python 3.

The main Python libraries used include:

* NumPy
* Pandas
* Matplotlib
* SciPy
* scikit-learn
* scikit-image
* LIME
* SHAP
* DiCE (`dice-ml`)

## Running the Experiments

The individual Python scripts can be executed separately depending on the required experiment.

For example:

```bash
python <script_name>.py
```

## Purpose

The purpose of this repository is to provide the implementation used to investigate and compare the behavior of LIME, SHAP, and Counterfactual Explanations across controlled synthetic classification scenarios and an FCC-based fringe classification use case.
