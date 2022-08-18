# Dependencies
import numpy as np
import pandas as pd
from flask import Flask, render_template, request,redirect

# Create an instance of Flask
app = Flask(__name__)

# =========================================================================================================
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

N_SAMPLES = 1000

classifiers = {
    "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),
    "Random Forest": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1
    ),
}

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)




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
    
    numwanting = request.form['numwanting']
    siteviews = request.form['siteviews']
    blogs = request.form['blogs']
    minage = request.form['minage']
    news = request.form['news']
    podcast = request.form['podcast']
    totalvotes = request.form['totalvotes']
    numcomments = request.form['numcomments']
    numgeeklists = request.form['numgeeklists']
    weblink = request.form['weblink']

    X_pred = np.array([[numwanting,siteviews, blogs, minage, news,
                podcast, totalvotes, numcomments, numgeeklists, weblink]])

    y_pred = rf.predict(X_pred)

    RF_pred = [round(y_pred[0], 2)]
    print(f'RF prediction= {RF_pred[0]}')
    print(X_pred[0])
    
    return render_template("index.html", RF_pred=RF_pred[0], X_pred=X_pred[0])

    
if __name__ == "__main__":
    app.run(debug=True)