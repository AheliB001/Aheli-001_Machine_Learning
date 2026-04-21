import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

data=load_diabetes()
x=data.data
y=data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
trees=[]
n_trees=5
for i in range(n_trees):
    tree=DecisionTreeRegressor(max_depth=2)
    tree.fit(x_train,y_train)
    trees.append(tree)

predictions=[]
for tree in trees:
    pred=tree.predict(x_test)
    predictions.append(pred)
predictions=np.array(predictions)
final_pred=np.mean(predictions, axis=0)
print(final_pred)
