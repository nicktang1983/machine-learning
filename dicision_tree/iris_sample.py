from sklearn import datasets
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot 

#digits = datasets.load_digits()
iris = datasets.load_iris()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
print clf.score(iris.data, iris.target)
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

dot_data = StringIO() 
#tree.export_graphviz(clf, out_file=dot_data) 

tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=iris.feature_names)  

graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 

