
from sklearn import svm
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
print(type(iris))
print(type(iris.data))
print(iris.feature_names)
print(iris.target_names)
x=iris.data[:,2]
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

model=svm.SVC(kernel='linear')
model.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))
y_predic=model.predict(x_test.reshape(-1,1))
print(metrics.accuracy_score(y_test.reshape(-1,1),y_predic.reshape(-1,1)
                             ))
