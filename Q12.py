import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("wine_train.csv", header = 0)
#xtest = pd.read_csv("/data/test/wine_test.csv", header = 0)
#xval = pd.read_csv("/data/test/wine_test.csv", header = 0)
xtrain,xtest = train_test_split(data,test_size=0.2)

#Write your code here
X_train = xtrain.drop('quality',axis=1)
y_train = xtrain['quality']
X_test = xtest.drop('quality',axis=1)
y_test = xtest['quality']
#wine_id = xtest['id']
print (X_train.shape,X_test.shape)
#X_val = xval.drop(['id','quality'],axis=1)
#y_val = xval['quality']

#lr = linear_model.LogisticRegression()
#lr.fit(X_train,y_train)
#y_pred = lr.predict(X_test)

#rf = GradientBoostingClassifier(n_estimators=300,random_state=2020)
rf = DecisionTreeClassifier()
param = {
    'n_estimators': [400,500],
     'max_depth' : [4,8],
      'criterion' :['gini', 'entropy'],
       'max_features':['auto', 'sqrt', 'log2']

}
#grid_rf = GridSearchCV(estimator=rf, param_grid=param)
#grid_rf.fit(X_train, y_train)

rf.fit(X_train, y_train)
y = rf.predict(X_train)
acc = metrics.accuracy_score(y_train,y)
print (acc)

y_pred = rf.predict(X_test)
acc = metrics.accuracy_score(y_test,y_pred)
print (acc)
#f1 = metrics.f1_score(y_test,y_pred)
#print (f1)

category = []
for num in y_pred:
    if num < 6:
        category.append("bad")
    elif num > 6:
        category.append("good")
    else:
        category.append("normal")

#finalOutput = pd.DataFrame({'id': wine_id,'taste': category})

#finalOutput.columns = ['id', 'taste']
#finalOutput.to_csv("/code/wine_prediction.csv", index = False)