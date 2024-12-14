import numpy as np 
import pandas as pd 

baseDeDados = pd.read_csv('admission.csv', delimiter=';')
X = baseDeDados.iloc[:,:-1].values
y = baseDeDados.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit_transform(X[:,1:])

# Codificando os rótulos (Label Encoding)
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

X = X[:,1:]
D = pd.get_dummies(X[:,0], dtype=int)
X = np.insert(X, 0, D.values, axis=1)

from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

print(XTrain)