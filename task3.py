# Import SHAP
import shap

# Define prediction function for H2O model
def h2o_predict(data_as_np_array):
    # Ensure the input matches the expected format
    h2o_frame = h2o.H2OFrame(data_as_np_array, column_names=X_train.columns.tolist())
    predictions = best_model.predict(h2o_frame).as_data_frame(use_multi_thread=True).values.flatten()
    return predictions

# Initialize SHAP Kernel Explainer using a sample of the training data
explainer = shap.KernelExplainer(h2o_predict, X_train.sample(100).values)  # Use a sample of the training data

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test.values)

# Visualize SHAP values
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist())
