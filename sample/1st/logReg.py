import sys
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import csv

# python logReg.py feature-train.csv feature-test.csv out.csv
X=[]
Y=[]
L=[]

# load features of training data
trainf=open(sys.argv[1])
for sl in csv.reader(trainf):
    L.append(sl[0])
    Y.append(int(sl[1]))
    xx=map(float,sl[2:])
    X.append(xx)

# set up training model
clf = LogisticRegression(penalty='l1',tol=0.01)
# training
clf = clf.fit(X, Y)
print clf.score(X,Y)
print clf.coef_

del X

# save model into pickle file
pf=open('clf-logReg.pkl','w')
s = pickle.dump(clf, pf)
pf.close()

X=[]
L=[]
# load features of testing data
testf=open(sys.argv[2])
for sl in csv.reader(testf):
    L.append(sl[0])
    xx=map(float,sl[1:])
    X.append(xx)

# make prediction
Y=clf.predict_proba(X)

# write result into csv
outf=open(sys.argv[3], 'w')
for i in xrange(0, len(Y)): 
    outs='%s,%s\n'%(L[i], ','.join(['%.4f'%y for y in Y[i]]))
    outf.write(outs)

