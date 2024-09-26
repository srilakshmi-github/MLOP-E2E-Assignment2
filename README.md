# MLOP-E2E-Assignment2
End-to-End Machine Learning Workflow

To execute the code on your local machine:

1.Update the model save path: Ensure the file path for saving the trained model is 
  correctly set according to your directory structure.
  
2.Save Task 4 in app.py: Move the code for Task 4 into a file named app.py.

3.Run the application: Use the following command in your terminal to start the app:-python app.py

4.Make a prediction: Use this curl command to send a POST request with the required data:
   curl -X POST -H "Content-Type: application/json" -d '{"Age": 36, "Salary": 65000, "Gender": "Female", "Department": "Research"}' 
   http://127.0.0.1:5000/predict
