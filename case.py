import pandas as pd

data=pd.read_csv(r"C:\Users\johne\OneDrive\Desktop\ICTAK data\Social_Network_Ads.csv")
dt=data.copy()

dt.drop('User ID',axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

dt['Gender']=le.fit_transform(dt['Gender'])

x=dt.drop('Purchased',axis=1)
y=dt['Purchased']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=24,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

rfc_model=rfc.fit(x_train,y_train)
rfc_pred=rfc_model.predict(x_test)

import pickle
with open('model2.pkl','wb') as file:
    pickle.dump(rfc_model,file)