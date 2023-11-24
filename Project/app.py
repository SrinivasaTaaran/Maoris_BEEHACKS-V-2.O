from flask import Flask, render_template, request
import pickle
import numpy as np

# loading the label encoder 
#le=pickle.load(open('label_encoder.pkl','rb'))

# loading my mlr model
model = pickle.load(open('model.pkl', 'rb'))

# loading Scaler
scalar = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/pred', methods=['POST'])
def predict1():
    rd = request.form["R&D Spend"]
    ad = request.form["Administration"]
    ms = request.form["Marketing Spend"]
    s = request.form["State"]
    t = [[float(rd), float(ad), float(ms), float(s)]]
    x = scalar.transform(t)
    output = model.predict(x)
    print(output)
    
    return render_template("home.html", result="The predicted profit is " + str(np.round(output[0])))

if __name__ == "__main__":
    app.run(debug=True)
