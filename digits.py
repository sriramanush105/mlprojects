from sklearn import datasets,svm
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
digits=datasets.load_digits()
print(digits.data.shape)
x=digits.data
y=digits.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=5)
clf1=LinearRegression()
clf2=LogisticRegression()
clf3=svm.SVC()
clf1.fit(x_train,y_train)
clf2.fit(x_train,y_train)
clf3.fit(x_train,y_train)
y_predic1=clf1.predict(x_test)
y_predic2=clf2.predict(x_test)
y_predic3=clf3.predict(x_test)
print("Linear Regression accuracy:",r2_score(y_predic1,y_test))
print("Logistic Regression accuracy:",r2_score(y_predic2,y_test))
print("support vector classifier accuracy:",r2_score(y_predic3,y_test))
