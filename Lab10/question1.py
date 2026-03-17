import pandas as pd
import math
from sklearn.model_selection import train_test_split

data = pd.read_csv('sonar_data.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

def entropy(labels):
    total = len(labels)              #count total number of data points
    counts={}
    for label in labels:
        if label in counts:          #how many times each label appears
         counts[label]+=1
        else:
         counts[label]=1
    ent=0                            #start value of entropy
    for count in counts.values():
        p= count/total               #to calculate probability
        ent = ent - p*math.log(p,2)
    return ent

entropy_value=entropy(list(y_test))
print(entropy_value)