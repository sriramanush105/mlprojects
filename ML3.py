from sklearn import datasets
from sklearn import svm
digits=datasets.load_digits()

n_samples=len(digits.images)
print(digits.images.shape)

data=digits.images.reshape(n_samples,-1)
classifier=svm.SVC(gamma=0.001)
classifier.fit(data,digits.target)
print(classifier.predict(digits.images[21].reshape(1,-1)))

