# Dependencies
import numpy as np
import pandas as pd
from flask import Flask, render_template, request,redirect

# Create an instance of Flask
app = Flask(__name__)

# =========================================================================================================

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def home():
    print(f"home")

    # Return template and data
    return render_template("index.html")

    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)