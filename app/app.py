# Dependencies
import numpy as np
import pandas as pd
from flask import Flask, render_template, request,redirect
import pickle

# Create an instance of Flask
# app = Flask(__name__)
def get_prediction(score):
    '''
    score float: model proba
    return str: Legit or Fraud
    '''
    return 'Fraud' if score >=0.8 else 'Legit'

# load trained classifier
model_path = 'models/model_rf8.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# =========================================================================================================

app = Flask(__name__, static_url_path='/static')

RF_pred="  "
X_pred=[]

@app.route("/")
def home():
    print(f"home")

    # Return template and data
    return render_template("index.html", RF_pred=RF_pred[0],X_pred=X_pred)

@app.route("/prediction", methods=['POST','GET'])
def prediction():
    print(f"prediction")
    
    MaxHospitalDays = request.form['MaxHospitalDays']
    TotalInscClaimAmtReimbursed = request.form['TotalInscClaimAmtReimbursed']
    TotalIPAnnualReimbursementAmt = request.form['TotalIPAnnualReimbursementAmt']
    MaxInscClaimAmtReimbursed = request.form['MaxInscClaimAmtReimbursed']
    MaxDiagCodeNumPerClaim = request.form['MaxDiagCodeNumPerClaim']
    TotalDiagCodeNum = request.form['TotalDiagCodeNum']
    MaxProcCodeNumPerClaim = request.form['MaxProcCodeNumPerClaim']
    MeanProcCodeNumPerClaim = request.form['MeanProcCodeNumPerClaim']

    X_pred = pd.Series([MaxHospitalDays,TotalInscClaimAmtReimbursed,TotalIPAnnualReimbursementAmt,MaxInscClaimAmtReimbursed,
                MaxDiagCodeNumPerClaim,TotalDiagCodeNum,MaxProcCodeNumPerClaim,MeanProcCodeNumPerClaim],
                index=['MaxHospitalDays', 'TotalInscClaimAmtReimbursed', 'TotalIPAnnualReimbursementAmt', 'MaxInscClaimAmtReimbursed',
                    'MaxDiagCodeNumPerClaim', 'TotalDiagCodeNum', 'MaxProcCodeNumPerClaim', 'MeanProcCodeNumPerClaim'])
    print(X_pred)

    y_pred = model.predict(X_pred.to_dict())
    print(y_pred)
    RF_pred = y_pred[:,1]
    print(f'RF prediction= {RF_pred[0]}')
    print(X_pred.to_list())
    
    return render_template("index.html", RF_pred=get_prediction(RF_pred[0]), X_pred=X_pred)

    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)