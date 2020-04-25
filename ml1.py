from sklearn import tree
features=[[150,1],[140,1],[180,1],[120,0],[200,0],[220,0]]
labels=[0,0,0,1,1,1]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print(clf.predict([[200,0]]))
      
