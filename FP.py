import numpy as np
from numpy import genfromtxt
import sklearn as sk
import sklearn.neighbors as skn
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import svm

my_data = genfromtxt('RS1.csv', delimiter=',',dtype=str)
risklabels = ('Low Risk','Moderate Risk','High Risk','No Risk')
InspectType = ('Routine - Unscheduled','New Ownership')
val = my_data[:,[2]]
# labels = my_data[:,[4]][1:]
traininglabels = []
for p in range(0,17000):
    traininglabels.append(my_data[p][4])
# print(np.alen(traininglabels))
testlabels = []
for p in range(17000,np.alen(my_data)-1):
    # print(p)
    testlabels.append(my_data[p][4])
voilation_id = {}
list_of_voilationids = []
h = 4
for k in val:
     ls = k[0].split('_')
     if voilation_id.get(ls[2]) == None:
        voilation_id[ls[2]] = h
        h = h+1
     list_of_voilationids.append(ls[2])
finalmatrix = np.zeros((np.alen(my_data),72),dtype = float)
for dt in range(0,np.alen(my_data)-1):
    if my_data[dt][0] == 'Low Risk':
        finalmatrix[dt][0] = 0
    if my_data[dt][0] == 'Moderate Risk':
        finalmatrix[dt][0] = 1
    if my_data[dt][0] == 'High Risk':
        finalmatrix[dt][0] = 2
    if my_data[dt][0] == 'No Risk':
        finalmatrix[dt][0] = 3
    finalmatrix[dt][2] = my_data[dt][5]
    finalmatrix[dt][3] = my_data[dt][6]
 # finalmatrix[dt][0] = my_data[dt][0]
    if my_data[dt][1] == 'New Ownership':
     finalmatrix[dt][1] = 1
    finalmatrix[dt][voilation_id.get(list_of_voilationids[dt-1])] = 1
minimum = finalmatrix[:,[0]]
min5 = finalmatrix[:,[2]]
min6 = finalmatrix[:,[3]]
minimu = np.min(minimum)
maxima = np.max(minimum)
minimu5 = np.min(min5)
maxima5 = np.max(min5)
minimu6 = np.min(min6)
maxima6 = np.max(min6)
for hp in range(0,np.alen(finalmatrix)):
     finalmatrix[hp][0] = np.divide(finalmatrix[hp][0] - minimu, maxima - minimu)
     finalmatrix[hp][2] = np.divide(finalmatrix[hp][2] - minimu5, maxima5 - minimu5)
     finalmatrix[hp][3] = np.divide(finalmatrix[hp][3] - minimu6, maxima6 - minimu6)
# finalmatrix = finalmatrix[1:]
trainingset = finalmatrix[:17000]
# print(np.alen(finalmatrix))
# print(np.alen(my_data))
# print(np.alen(trainingset))
# KNN Classifier
testset = finalmatrix[17000:np.alen(my_data)-1]
knn = skn.KNeighborsClassifier(n_neighbors = 1)
knn.fit(trainingset, traininglabels)
pred = knn.predict(testset)
# print(testlabels)
# print(pred)
accuracy = sk.metrics.accuracy_score(testlabels, pred)
# print(accuracy*100)
# plt.scatter(testset[:,[0]],trainingset[:,[0]])
# plt.show()
# print(accu*100)
# print(testlabels)
# print(pred)
# print(trainingset[16999])
# Naive Bayes Classifier
model = GaussianNB()
model.fit(trainingset,traininglabels)
predicted = model.predict(testset)
accuracy = sk.metrics.accuracy_score(testlabels,predicted)
print(accuracy)

# Logistic Regression
clf = LogisticRegression(random_state=0,solver='newton-cg',multi_class='multinomial').fit(trainingset,traininglabels)
pr = clf.predict(testset)
accuracy = sk.metrics.accuracy_score(testlabels,pr)
print(pr)
print(testlabels)
print(accuracy*100)


clf = svm.SVC(gamma='scale',decision_function_shape='ovo')
clf.fit(trainingset,traininglabels)
pred = clf.predict(testset)
accuracy = sk.metrics.accuracy_score(testlabels,pred)
print(accuracy*100)