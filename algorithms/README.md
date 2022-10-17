# Explainable AI

## Introduction
Machine learning algorithms' output and outcomes may now be understood and trusted by human users thanks to a collection of procedures and techniques known as explainable artificial intelligence (XAI). An AI model, its anticipated effects, and potential biases are all described in terms of explainable AI. It contributes to defining model correctness, fairness, transparency, and results in decision-making supported by AI. When bringing AI models into production, a business must first establish trust and confidence. A company may adopt a responsible approach to AI development with the aid of AI explainability.

## Different algorithms

### LIME (Local Interpretable Model-agnostic Explanations)

Local Interpretable Model-agnostic Explanations is the acronym for LIME. It is a method of visualization that aids in the explanation of certain predictions. It may be used with any supervised regression or classification model because it is model independent (explaining the working of machine learning and deep learning models). Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin presented LIME in 2016. It is applicable to both organized and unstructured information, including text and picture data. The official [repository](https://github.com/marcotcr/lime) supports the [documentation](https://lime-ml.readthedocs.io/en/latest/lime.html#) for understanding the different modules it provides.

Black box machine learning models are used to explain specific predictions using local surrogate models, which are interpretable models. In the above study Local interpretable model-agnostic explanations (LIME), the authors suggest a practical use of local surrogate models. In order to approximate the predictions of the underlying black box model, substitute models are developed. LIME focuses on training local surrogate models to explain specific predictions rather than creating a global surrogate model. LIME operates on the presumption that all complicated models are linear on a local scale. LIME attempts to fit a straightforward model to a single observation that will replicate the local behavior of the global model.

By examining the internal workings and interactions of the black-box machine learning models, model-specific approaches seek to comprehend it. Local model interpretability is offered by LIME. LIME adjusts the feature values for a single data sample and tracks the effect on the output. What factors led to this forecast or why was it made is likely the most frequently asked question.

The predictions of the more sophisticated model may then be locally explained using the basic model. Three different input formats are supported by LIME: tabular data, text data, and picture data.


#### Intro to surrogate models

Local surrogate models with an interpretability restriction can be represented as follows:

$$ explanation(x) = {\arg \underset{g∈G}minL(f,g, π_{x})+Ω_{g}} $$

The model g (for example, a linear regression model) that minimizes loss L (for example, mean squared error), which assesses how closely the explanation matches the prediction of the original model f (for example, an xgboost model), is the explanation model for example x. Model complexity is also kept to a minimum (e.g. prefer fewer features). The family of explanations known as G includes, for instance, all potential linear regression models. The proximity measure specifies the size of the area surrounding instance x that we take into account while providing an explanation. In reality, LIME just improves the loss component. The user must choose the maximum amount of features the linear regression model may employ, for example, in order to define the complexity.

#### How LIME works

- To explain the observation, disrupt (unset or disturb) it n times to provide duplicated feature data with minimal value changes. The false data that has been generated around the observation for LIME to employ in creating the local linear model is the perturbed data.
- Forecast the results of the perturbed data.
- Determine the separation between each disturbed data point and the initial observation.
- The distance has to be calculated based on a similarity score
- Choose m features from the perturbed data that best represent the predictions.
- To account for the chosen characteristics in the perturbed data, fit a basic model.
- The basic model's feature weights (coefficients) provide as an explanation for the observation.

With simple words, LIME investigates what happens to the predictions when we alter the data that the machine learning model is fed. So, LIME creates a new dataset made up of converted samples and the matching black-box model predictions. Based on the new dataset, a new model is trained which is interpretable.

The aim is to comprehend why a certain prediction was produced by the machine learning model. When different versions of the data are fed into the machine learning model, LIME examines what happens to the predictions. LIME creates the related black box model predictions with the use of the brand-new dataset. The weighting of the interpretable model that LIME trains on this new dataset is based on how close the sampled examples are to the instance of interest. Any model from the chapter on interpretable models, such as Lasso or a decision tree, may be used as the interpretable model. Locally, the learnt model should be a good approximation of the predictions made by the machine learning model; however, a good global approximation is not required.

Locally, the learned model should be a good approximation of the predictions made by the machine learning model; however, a good global approximation is not required. Another name for this level of accuracy is "local fidelity".

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
["Interpretable machine learning](https://christophm.github.io/interpretable-ml-book/)
