from flask import Flask, request, jsonify
import pandas as pd
import h5py
import pickle
import base64

with h5py.File('model.h5', 'r') as f:
    model_string = f['model'][()]
    model_M = pickle.loads(base64.b64decode(model_string))

# Create a Flask app
app = Flask(__name__)

@app.route("/predict/<string:one>/<string:two>/<string:three>/<string:four>/<string:five>/<string:six>/<string:seven>/<string:eight>", methods=["GET"])
def predict(one, two, three, four, five, six, seven, eight):
    try:
        data = [float(one), float(two), float(three), float(four), float(five), float(six), float(seven), float(eight)]
        res = model_M.predict([data]).tolist()  
        return jsonify({"Prediction": res})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()
