from flask import Flask, request, jsonify
import h2o
import joblib

# Initialize H2O
h2o.init()

# Load the model
model_path = '/your_model_file.zip/GBM_grid_1_AutoML_27_20240922_223833_model_1'  # Update this path to your saved model
best_model = h2o.load_model(model_path)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json(force=True)
    
    # Convert data to H2OFrame
    h2o_frame = h2o.H2OFrame(data)
    
    # Make predictions
    predictions = best_model.predict(h2o_frame).as_data_frame().values.flatten().tolist()
    
    # Return the predictions
    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
