import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import pickle
from model_files.model import make_recommendation


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

    beer = request.get_json()

    print(beer)

    beer_recommendation = make_recommendation(beer, model) 

    output = print(jsonify(beer_recommendation))

    return render_template('index.html', prediction_text= format(output))


@app.route("/beermap")
def beermap():
    return render_template("beermap.html")


if __name__ == '__main__':
    app.run(debug=True)