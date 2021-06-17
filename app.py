import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import pickle

#flask setup
from flask import Flask, jsonify, render_template, request, flash, redirect
app = Flask(__name__, template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    #scrape_dict = collection.find_one()
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    

    output = final_features[0]

    return render_template('index.html', prediction_text= format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route("/beermap")
def beermap():
    return render_template("beermap.html")


if __name__ == '__main__':
    app.run(debug=True)