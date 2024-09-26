# Import necessary libraries for H2O and joblib
import h2o
from h2o.automl import H2OAutoML
import pandas as pd  # Make sure to import pandas if not already done

# Initialize H2O server
h2o.init()

# Convert Pandas DataFrames to H2O Frames
train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
val_h2o = h2o.H2OFrame(pd.concat([X_val, y_val], axis=1))

# Define target and features
target = 'Age'
features = X_train.columns.tolist()

# AutoML model training
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=features, y=target, training_frame=train_h2o, validation_frame=val_h2o)

# Leaderboard and best model
lb = aml.leaderboard
print(lb)

# Model performance on validation data
best_model = aml.leader
performance = best_model.model_performance(val_h2o)
print(performance)

# Save the trained model
model_path = 'your_model_file.zip'  # Specify your model's save path
h2o.save_model(model=best_model, path=model_path, force=True)  # Save the model
print(f'Model saved to: {model_path}')
