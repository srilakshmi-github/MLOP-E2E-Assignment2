from flask import Flask, request, jsonify
import h2o
import joblib

# Initialize the H2O server
h2o.init()

# Load the pre-trained model
model_path = '/your_model_file.zip/GBM_grid_1_AutoML_27_20240922_223833_model_1'  # Update this path with your model's location
best_model = h2o.load_model(model_path)

# Create a Flask web application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the JSON data from the request
    data = request.get_json(force=True)
    
    # Convert the incoming data into an H2OFrame
    h2o_frame = h2o.H2OFrame(data)
    
    # Generate predictions using the model
    predictions = best_model.predict(h2o_frame).as_data_frame().values.flatten().tolist()
    
    # Send the predictions back as a JSON response
    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode
