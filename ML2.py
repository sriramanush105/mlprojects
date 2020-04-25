from sklearn import svm
from matplotlib import image
import os
import numpy as np
features=list()
for f in os.listdir(r'C:\Users\Sriram\Desktop\300'):
    if f.endswith('.png'):
        try:
            img=image.imread(r'C:\Users\Sriram\Desktop\300/'+f)
            print(img)
            print(type(img))
            print(img.shape)
          
            features.append(img.reshape(img.shape[0],-1))
            print(" hee")
        except:
            pass
      
for f in os.listdir(r'C:\Users\Sriram\Desktop\300n'):
    
    if f.endswith('.png'):

        
        try:
            img=image.imread(r'C:\Users\Sriram\Desktop\300n/'+f)
            features.append(img.reshape(img.shape[0],-1))
        except:

            pass
features=np.array(features)
m=len(features)
data=features.reshape(m,-1)
l=[0]
p=[1]
print("hello")
print(data.shape)
targets=l*20+p*20
targets=np.array(targets)
clf=svm.SVC(gamma=0.001)
clf=clf.fit(features,targets)
clf.predict(features[4].reshape(1,-1))
