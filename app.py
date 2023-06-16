from flask import Flask, jsonify, request
import pickle
from pandas import DataFrame 

# Load the trained model
model = pickle.load(open("model_v1.pkl", "rb"))

# Create the Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "home page, welcome dump."

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()
    
    # Extract input features
    weight = data['weight']
    height = data['height']
    bmi = data['bmi']
    age = data['age']
    gender = data['gender']
    fat_percent = data['fat_percent']
    data ={
    "weight": [weight],
    "height": [height],
    "bmi":[bmi],
    "age":[age],
    "gender":[gender],
    "fat_percent":[fat_percent]
    }
    data = DataFrame(data)
    
    # Make a prediction using the loaded model
    # prediction = model.predict([[weight, height, bmi, age, gender, fat_percent]])
    prediction = model.predict(data)
    
    # Map the prediction to the corresponding class
    classes = ['Over Weight', 'Obese', 'Under Weight', 'Healthy Weight']
    predicted_class = classes[prediction[0]]
    # Return the prediction as a JSON response
    response = {"prediction": predicted_class}
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000,debug=True)
