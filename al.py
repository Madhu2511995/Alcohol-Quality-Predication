import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")

alcohol=pd.read_csv('alcohol-quality-data.csv')



# Convert the Object value into the measure value
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
alcohol['Quality_Category']=le.fit_transform(alcohol['Quality_Category'])


l=['pH','sulphates','alcohol']
X=alcohol[l]
y=alcohol['Quality_Category']



'''from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)'''

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X,y)
print(lr.score(X,y))

pickle.dump(lr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[3,0.49,9.5]]))
