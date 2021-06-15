import pandas as pd
from sqlalchemy import create_engine

#flask setup
from flask import Flask, jsonify, render_template, request, flash, redirect
app = Flask(__name__)


@app.route("/")
def home():
    #scrape_dict = collection.find_one()
    return render_template("index.html")

@app.route("/beermap")
def beermap():
    return render_template("beermap.html")


if __name__ == '__main__':
    app.run(debug=True)