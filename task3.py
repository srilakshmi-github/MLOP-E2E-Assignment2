# Import SHAP for explainable AI
import shap

# Define a prediction function for the H2O model
def h2o_predict(data_as_numpy):
    # Convert the input data into H2OFrame format
    h2o_frame = h2o.H2OFrame(data_as_numpy, column_names=X_train.columns.tolist())
    # Get predictions from the best model and return them as a flattened array
    predictions = best_model.predict(h2o_frame).as_data_frame(use_multi_thread=True).values.flatten()
    return predictions

# Initialize SHAP Kernel Explainer using a sample of the training data
explainer = shap.KernelExplainer(h2o_predict, X_train.sample(100).values)  # Use a subset of the training data

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test.values)

# Display SHAP summary plot for interpretability
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist())
