from sklearn import datasets
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot 

digits = datasets.load_digits()
#digits = datasets.load_iris()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(digits.data, digits.target)

with open("digits.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("digits.pdf") 

