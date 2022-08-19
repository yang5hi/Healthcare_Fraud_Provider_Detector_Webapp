from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import ToyModel
from transformer import Cost_Transformer
import os
import json

app = Flask(__name__)
api = Api(app)

# load trained classifier
model_path = 'models/model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

def get_prediction(score):
    '''
    score float: model proba
    return str: negative or positive
    '''
    return 'Positive' if score >=0.5 else 'Negative'

class PredictToy(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        # json needs to replace single quote with double 
        predict_proba = model.predict(json.loads(user_query.replace("\'", "\"")))
        results = {'results':[]}
        for proba in predict_proba[:,1]:
            results['results'].append({'label': get_prediction(proba), 'ModelScore':proba})      
        return results


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictToy, '/')


if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0", port= int(os.environ.get("PORT", 5000)))