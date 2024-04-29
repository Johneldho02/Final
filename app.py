from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict():
    with open('model3.pkl','rb') as file:
        model=pickle.load(file)    
    if request.method == 'POST':
        ge = str(request.form['gen'])
        ag = int(request.form['age'])
        est = int(request.form['est_sal'])
    if ge=='Male':
        genn=1
    else:
        genn=0   

    feature=np.array([[genn,ag,est]])
    ans=np.array(['No, the given customer will not make a purchase :(','Yes, the given customer will make a purchase :)'])
    prediction = model.predict(feature)
    final=ans[prediction[0]]

    return render_template('prediction.html',final_=final)    

if __name__=='__main__':
    app.run()