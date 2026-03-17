import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data=pd.read_csv('sonar_data.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

#convert r and m to numbers
encoder=LabelEncoder() #sorts them alphabetically
y=encoder.fit_transform(y)
print(encoder.classes_) #TO CHECK MAPPING. M IS ASSIGNED 0 AND R 1
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=180)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy: ",accuracy)
