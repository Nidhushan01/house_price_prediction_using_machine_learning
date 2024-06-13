from flask import Flask, request, render_template, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model

    with open(r"C:\Users\ASUS\Desktop\home price\model\columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    with open(r"C:\Users\ASUS\Desktop\home price\model\banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', locations=__locations)

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    estimated_price = get_estimated_price(location, sqft, bhk, bath)
    return jsonify({'estimated_price': estimated_price})

if __name__ == '__main__':
    load_saved_artifacts()
    app.run(debug=True)
