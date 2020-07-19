from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the LogisticRegression model and Tfidfvectorizetion object from disk
filename = 'CovidSpam-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
tf = pickle.load(open('tf-transform.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/Predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tf.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('index.html', prediction=my_prediction)
    
@app.route("/prediction", methods=['GET','POST'])   
def prediction():
    if request.method == 'POST':
        try:
            Age = float(request.form['age'])
            Gender = float(request.form['gender'])
            Body_Temperature = float(request.form['bodytemperature'])
            DryCough = float(request.form['DryCough'])
            Sour_Throat = float(request.form['sourthroat'])
            Weakness = float(request.form['weakness'])
            Breathing_Problem = float(request.form['breathingproblem'])
            Diabetes = float(request.form['diabetes'])
            Drowsiness = float(request.form['drowsiness'])
            Travel_History = float(request.form['travelhistory'])
            pred_args = [Age, Gender, Body_Temperature, DryCough, Sour_Throat, Weakness,
                         Breathing_Problem, Diabetes, Drowsiness, Travel_History]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            Covid_Symptoms = open("CovidSymtoms.pkl", "rb")
            ml_model = pickle.load(Covid_Symptoms)
            model_prediction = ml_model.predict_proba(pred_args_arr)
            Low = model_prediction[0][0]
            Medium = model_prediction[0][1]
            High = model_prediction[0][2]
        except ValueError:
            return "Please Check if values are written correctly"
    return render_template('predict.html', Low=Low, Medium=Medium, High=High)

if __name__ == '__main__':
    app.run(debug=True)