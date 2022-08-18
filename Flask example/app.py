from flask import Flask, render_template, request
num_3=""
app = Flask(__name__)

@app.route("/", methods =["GET", "POST"])
def index():
	result="Classification Model Comparison"
	num_2 = request.form['num_2']
	num_3=int(num_2)+5

	return render_template("index.html", result=result,num_3=num_3)

if __name__== "__main__":
	app.run(debug=True, host='0.0.0.0', port=5000)