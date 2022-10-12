# Explainable AI

## Introduction
Machine learning algorithms' output and outcomes may now be understood and trusted by human users thanks to a collection of procedures and techniques known as explainable artificial intelligence (XAI). An AI model, its anticipated effects, and potential biases are all described in terms of explainable AI. It contributes to defining model correctness, fairness, transparency, and results in decision-making supported by AI. When bringing AI models into production, a business must first establish trust and confidence. A company may adopt a responsible approach to AI development with the aid of AI explainability.

## Different algorithms

### LIME (Local Interpretable Model-agnostic Explanations)

Local Interpretable Model-agnostic Explanations is the acronym for LIME. It is a method of visualization that aids in the explanation of certain predictions. It may be used with any supervised regression or classification model because it is model independent (explaining the working of machine learning and deep learning models). Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin presented LIME in 2016. It is applicable to both organized and unstructured information, including text and picture data. The official [repository](https://github.com/marcotcr/lime) supports the [documentation](https://lime-ml.readthedocs.io/en/latest/lime.html#) for understanding the different modules it provides.

LIME operates on the presumption that all complicated models are linear on a local scale. LIME attempts to fit a straightforward model to a single observation that will replicate the local behavior of the global model.

The predictions of the more sophisticated model may then be locally explained using the basic model. Three different input formats are supported by LIME: tabular data, text data, and picture data.

#### How LIME works

- To explain the observation, disrupt (unset or disturb) it n times to provide duplicated feature data with minimal value changes. The false data that has been generated around the observation for LIME to employ in creating the local linear model is the perturbed data.
- Forecast the results of the perturbed data.
- Determine the separation between each disturbed data point and the initial observation.
- The distance has to be calculated based on a similarity score
- Choose m features from the perturbed data that best represent the predictions.
- To account for the chosen characteristics in the perturbed data, fit a basic model.
- The basic model's feature weights (coefficients) provide as an explanation for the observation.

LIME treats continuous data differently than it does for categorical variables. The perturbed data for categorical variables will have binary values of 0 or 1. If the perturbed item matches the observation that has to be explained, the value will be 1, otherwise it will be 0.

#### Tuning Lime

Lime can be tuned for features:

• LIME builds a local linear model with the amount of features you provide.
• Depending on how complicated the model is and how straightforward an explanation is needed, the user can adjust the amount of features.

(methods that are supported for feature selection are: "Highest Weights", "Forward Selection", "Lasso Path", "none", "auto")
and for samples:

• LIME gives the user the option to fine-tune how many samples should be generated for the perturbed data. (5000 is the default)
• The parameter can be set to various sample sizes for example 1000, 5000, and 10000.

#### Creation of pertubed data

- LIME perturbs the input observation that needs to be explained in relation to its locale and generates a local data.
- Continual variables are perturbed by sampling from a Normal (0,1) distribution and performing the inverse operation of mean-centering and scaling, in accordance with the means and standard deviations in the training data. Categorical variables are given random values based on the possible category values and their frequency of occurrence in the training dataset.
- The observation that needs to be explained will always be in the first row of the perturbed data. Scaling is applied to continuous variables in the perturbed data.
- Perturbed data for categorical variables have values of 0 or 1. (data). 0 unless the category matches the observation to be explained, then 1.

#### Drawbacks of LIME

- It's possible that some "unlikely" data points are formed as LIME generates its own sample data based on the gaussian distribution around the location of the observation, from which the model may have learned.
- The linear model might be unable to adequately explain the decision boundary if it is too non-linear.


## References

["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
[Understanding lime from cran.r-project.org](https://cran.r-project.org/web/packages/lime/vignettes/Understanding_lime.html)
