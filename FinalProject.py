import numpy as np
from numpy import genfromtxt
import sklearn as sk
import sklearn.neighbors as skn
import matplotlib.pyplot as plt
my_data = genfromtxt('RS.csv', delimiter=',',dtype=str)
risklabels = ('Low Risk','Moderate Risk','High Risk','No Risk')
InspectType = ('Routine - Unscheduled','New Ownership')
val = my_data[:,[2]][1:]
labels = my_data[:,[4]][1:]
traininglabels = []
# print(np.alen(my_data))
for p in range(0,30000):
    if labels[p] == 'Low Risk':
        traininglabels.append(0)
    if labels[p] == 'Moderate Risk':
        traininglabels.append(1)
    if labels[p] == 'High Risk':
        traininglabels.append(2)
    if labels[p] == 'No Risk':
        traininglabels.append(3)
testlabels = []
for p in range(30000,np.alen(my_data)-1):
    if labels[p] == 'Low Risk':
        testlabels.append(0)
    if labels[p] == 'Moderate Risk':
        testlabels.append(1)
    if labels[p] == 'High Risk':
        testlabels.append(2)
    if labels[p] == 'No Risk':
        testlabels.append(3)
voilation_id = {}
list_of_voilationids = []
h = 2
for k in val:
     ls = k[0].split('_')
     if voilation_id.get(ls[2]) == None:
        voilation_id[ls[2]] = h
        h = h+1
     list_of_voilationids.append(ls[2])
finalmatrix = np.zeros((np.alen(my_data),70),dtype = float)
for dt in range(1,np.alen(my_data)):
 finalmatrix[dt][0] = my_data[dt][0]
 if my_data[dt][1] == 'New Ownership':
     finalmatrix[dt][1] = 1
 finalmatrix[dt][voilation_id.get(list_of_voilationids[dt-1])] = 1
minimum = finalmatrix[:,[0]][1:]
minimu = np.min(minimum)
maxima = np.max(minimum)
for hp in range(1,np.alen(finalmatrix)):
    finalmatrix[hp][0] = np.divide(finalmatrix[hp][0]-minimu,maxima-minimu)
finalmatrix = finalmatrix[1:]
trainingset = finalmatrix[:30000]
testset = finalmatrix[30000:np.alen(my_data)-1]
knn = skn.KNeighborsClassifier(n_neighbors=3)
knn.fit(trainingset, traininglabels)
pred = knn.predict(testset)
accuracy = sk.metrics.accuracy_score(testlabels, pred)
print(accuracy*100)
# plt.scatter(testset[:,[0]],trainingset[:,[0]])
# plt.show()
# print(accu*100)
# print(testlabels)
# print(pred)