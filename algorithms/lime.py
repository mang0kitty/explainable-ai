import lime
from lime import lime_tabular


def create_lime_text_explanation(
    train_data,
    model,
    feature_names,
    class_names,
    data_point,
    num_features=2,
    top_labels=1,
    show_all=True,
    discretize_continuous=True,
    show_table=True,
):
    """
    If show_all=False only the features used in the explanation are displayed.
    If discretize_continuous=True, LIME discretizes the features in the explanation. Discretized characteristics provide for better understandable explanations.
    If show_table=True, a table with features and values is also displayed
    """
    explainer = lime_tabular.LimeTabularExplainer(
        train_data,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=discretize_continuous,
    )
    explanation = explainer.explain_instance(
        data_point,
        model.predict_proba,
        num_features=num_features,
        top_labels=top_labels,
    )
    explanation.show_in_notebook(show_table=show_table, show_all=show_all)
