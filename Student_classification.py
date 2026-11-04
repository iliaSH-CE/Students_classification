#Used libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score


#Reading Data
data=pd.read_csv("File's name",delimiter=';')


#binerizing objective and categorial values with LableEncoder
for i in data.select_dtypes(include=['object','category']).columns:
        le=preprocessing.LabelEncoder()
        data[i]=le.fit_transform(data[i])


#
bins=[0,5,10,15,20]
lables=['Very bad','Bad','Average','Good']
data['Target_G3']=pd.cut(data['G3'],bins=bins,labels=lables,include_lowest=True)
le_target=preprocessing.LabelEncoder()
data['Target_G3']=le_target.fit_transform(data['Target_G3'])

#ّFeatures
X=data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences','G1', 'G2' ]]


#Target(G3=final Grade)
y=data[['Target_G3']]



#making train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)
#Using DecisionTreeClassifier as model
tree=DecisionTreeClassifier(max_depth=5,random_state=40).fit(X_train,y_train)
yhat=tree.predict(X_test)


#Evaluating the model
print("Acurracy:",metrics.accuracy_score(y_test,yhat))
print("Classification_report:",metrics.classification_report(y_test,yhat))
print("Confusion_Matrix:",metrics.confusion_matrix(y_test,yhat))
