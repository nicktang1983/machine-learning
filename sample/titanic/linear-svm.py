# example code for Titanic data on Kaggle
# https://www.kaggle.com/c/titanic

import csv as csv 
import numpy as np
from sklearn.svm import SVC

# read training csv file
csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()

x=[]
y=[]
lb=[]
for row in csv_file_object:
    # feature transform for train data
    xx = [0,0,0]

    # Pclass dummy variables
    xx[int(row[2])-1]=1
    
    # gender dummy variables
    if row[4]=='female': xx+=[0,1]
    else: xx+=[1,0]

    # age and fix missing value with avg age ~30
    try: xx+=[float(row[5])]
    except: xx+=[30]

    x.append(xx)
    if row[1]=='1': y.append(1)
    else: y.append(-1)
    lb.append(row[0])

# transform to numpy array used in sklearn
x=np.array(x)
y=np.array(y)

# model selection and parameter setting
clf = SVC(C=1.0, kernel='linear')

# train model
clf = clf.fit(x,y)
##print clf.score(x,y)

# read testing csv file
csv_file_object = csv.reader(open('test.csv', 'rb'))
header = csv_file_object.next()

x=[]
lb=[]
for row in csv_file_object:
    # feature transform for test data
    xx = [0,0,0]
    xx[int(row[1])-1]=1
    if row[3]=='female': xx+=[0,1]
    else: xx+=[1,0]
    try: xx+=[float(row[4])]
    except: xx+=[30]
    x.append(xx)
    lb.append(row[0])
##for i in xrange(len(x)): print lb[i], x[i]

# transform to numpy array used in sklearn
x=np.array(x)

# make prediction
y=clf.predict(x)

# output prediction result
print 'PassengerId,Survived'
for i in xrange(len(y)): 
    if y[i]>0: print '%s,1'%(lb[i]) 
    else: print '%s,0'%(lb[i]) 
