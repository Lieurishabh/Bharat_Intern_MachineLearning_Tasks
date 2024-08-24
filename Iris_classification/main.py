from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

data1 = pd.read_csv("final_iris_dataset.csv")
model = pickle.load(open("NBmodel.pkl", 'rb'))

app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        sl = request.form['SepalLengthCm']
        sw = request.form['SepalWidthCm']
        pl = request.form['PetalLengthCm']
        pw = request.form['PetalWidthCm']
        data = np.array([[sl, sw, pl, pw]], dtype=float)
       # x = scaler.transform(data)
        prediction = model.predict(data)
        if prediction[0] == 1:
            image = 'Iris-setosa.png'
        elif prediction[0] == 2:
            image = 'Iris-versicolor.png'
        elif prediction[0] == 3:
            image = 'Iris-virginica.png'
        prediction1 = image.split(".")
        prediction = str(prediction1[0])
        image = os.path.join(app.config['UPLOAD_FOLDER'], image)

        return render_template('index.html',prediction=prediction, image=image)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)