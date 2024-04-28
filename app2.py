from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)


with open('model2.pkl','rb') as file:
    model=pickle.load(file)

@app.route('/')   
def home():
    return render_template('index2.html')
@app.route('/predict',methods=['POST'])
def predict():
    gen = str(request.form['gen'])
    age = int(request.form['age'])
    est_sal = int(request.form['est_sal'])

    if gen=='Male':
        genn=1
    else:
        genn=0   
    feature=np.array([[genn,age,est_sal]])
    ans=np.array(['No','Yes'])
    prediction = model.predict(feature)
    final=ans[prediction[0]]
        
    return render_template('index2.html',pred_res=final) 
   
if __name__=='__main__':
    app.run(debug=True)





