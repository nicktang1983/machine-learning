# example code for Titanic data on Kaggle
# https://www.kaggle.com/c/titanic

import csv as csv 
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot 
# read training csv file
csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()
count=0

x=[]
y=[]
lb=[]

x_t=[]
y_t=[]
lb_t=[]

# for row in csv_file_object:
#     # feature transform for train data
#     xx = [0,0,0]

#     # Pclass dummy variables
#     xx[int(row[2])-1]=1
    
#     # gender dummy variables
#     if row[4]=='female': xx+=[0,1]
#     else: xx+=[1,0]

#     # age and fix missing value with avg age ~30
#     try: xx+=[float(row[5])]
#     except: xx+=[30]

#     if count < 500:
#         x.append(xx)
#         if row[1]=='1': y.append(1)
#         else: y.append(-1)
#         lb.append(row[0])
#     else:
#         x_t.append(xx)
#         if row[1]=='1': y_t.append(1)
#         else: y_t.append(-1)
#         lb_t.append(row[0])

#     count += 1


for row in csv_file_object:
    # feature transform for train data
    xx = [0]

    # Pclass dummy variables
    xx[0]=int(row[2])
    
    # gender dummy variables
    xx+=[row[4]]

    # age and fix missing value with avg age ~30
    try: xx+=[float(row[5])]
    except: xx+=[30]

    if count < 500:
        x.append(xx)
        if row[1]=='1': y.append(1)
        else: y.append(-1)
        lb.append(row[0])
    else:
        x_t.append(xx)
        if row[1]=='1': y_t.append(1)
        else: y_t.append(-1)
        lb_t.append(row[0])

    count += 1

# transform to numpy array used in sklearn
x=np.array(x)
y=np.array(y)
print x
print y
x_t=np.array(x_t)
y_t=np.array(y_t)

# model selection and parameter setting
 
# for i in range(5, 10):
#     # clf = SVC(C=1.0, kernel='linear')
#     clf = tree.DecisionTreeClassifier(max_depth=None)

#     # train model
#     clf = clf.fit(x,y)
#     print clf.score(x,y)
#     print clf.score(x_t, y_t)
#     print '*'*10

clf = tree.DecisionTreeClassifier(max_depth=None)
clf = clf.fit(x,y)
print clf.score(x,y)
print clf.score(x_t, y_t)
print '*'*10

with open("titanic.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

dot_data = StringIO() 
# feature_names = ['Pclass','Pclass2','Pclass2','male','female','Age']
feature_names = ['Pclass', 'Sex', 'Age']
tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("titanic.pdf") 

exit()

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
