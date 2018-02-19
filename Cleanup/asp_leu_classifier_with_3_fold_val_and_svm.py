import numpy as np
import numpy as np
import itertools
import sys
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

font = {'family' : 'Myriad Pro',
        'size'   : 18}
plt.rc('font', **font)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Differentiating Asp/Leu',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
 #   plt.colorbar()
    plt.clim((0,40))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

path = "feature_csvs_120s/" # path to feature csv's

count = int(sys.argv[1])

leu_1 = np.loadtxt(path + "100n_leu.csv", delimiter=",")
leu_2 = np.loadtxt(path + "10u_leu.csv", delimiter=",")
leu_3 = np.loadtxt(path + "1m_asp.csv", delimiter=",")

asp_1 = np.loadtxt(path + "100n_asp.csv", delimiter=",")
asp_2 = np.loadtxt(path + "10u_asp.csv", delimiter=",")
asp_3 = np.loadtxt(path + "1m_asp.csv", delimiter=",")

#shuffle
np.random.shuffle(leu_1[:,3])
np.random.shuffle(leu_2[:,3])
np.random.shuffle(leu_3[:,3])
np.random.shuffle(asp_1[:,3])
np.random.shuffle(asp_2[:,3])
np.random.shuffle(asp_3[:,3])

# biases, cw, ccw, switch
leu = iter(zip(leu_1[:,3], leu_2[:,3], leu_3[:,3]))
asp = iter(zip(asp_1[:,3], asp_2[:,3], asp_3[:,3]))

training_set = np.zeros((count, 4))

counter = 0
prev_count = counter
while counter < count:
    try:
        a, b, c = next(leu)
        training_set[counter] = np.asarray([a, b, c, 1])
        counter += 1
    except StopIteration:
        pass
    if counter < count:
        try:
            d, e, f = next(asp)
            training_set[counter] = np.asarray([d, e, f, -1])
            counter += 1
        except StopIteration:
            pass
    if prev_count == counter:
        break
    else:
        prev_count = counter

if counter != count:
    count = prev_count
    print("count is now: " + str(count))

folds = {0:training_set[0:count//3],1:training_set[count//3:2*count//3],2:training_set[2*count//3:count]}
error = 0
total_correct_train_per = 0
total_correct_val_per = 0
s = 0.0
cnf_matrix=np.zeros((2,2),np.int8)
for k in range(3):
    train = [0,1,2]
    train.remove(k)
    val = folds[k]
    t = np.concatenate((folds[train[0]],folds[train[1]]),axis=0)
    X = t[:,0:3]
    Y = t[:,-1]
    X_val = val[:,0:3]
    Y_val = val[:,-1]
    weights = np.linalg.lstsq(X,Y)[0]

    error += np.sqrt((np.linalg.norm(Y_val-np.dot(X_val,weights))/(count/3)))

    total_correct_train = 0
    total_correct_val = 0
    for i in range(0,2*count/3):
        entry = X[i,:]
        result = np.dot(entry,weights)
        if result <= 0 and Y[i] == -1:
            total_correct_train += 1
        elif result > 0 and Y[i] == 1:
            total_correct_train += 1
    total_correct_train_per += total_correct_train/(2*count/3.0)

    for i in range(0,count/3):
        entry = X_val[i,:]
        result = np.dot(entry,weights)
        if result <= 0 and Y_val[i] == -1:
            total_correct_val += 1
        elif result > 0 and Y_val[i] == 1:
            total_correct_val += 1
    total_correct_val_per += total_correct_val/(count/3.0)
    
    clf = svm.SVC(kernel='linear', C=1).fit(X,Y)
    s += clf.score(X_val, Y_val)
    Y_pred = clf.predict(X_val)
    cnf_matrix+=confusion_matrix(Y_val,Y_pred)

# plt.figure(figsize=(4,4))
# cnf_matrix[0,0] = 34
# cnf_matrix[0,1] = 11
# cnf_matrix[1,0] = 4
# cnf_matrix[0,0] = 41

#cnf_matrix = ((26,10,4),(8,26,6),(2,7,31))
plot_confusion_matrix(cnf_matrix,classes=['Leu','Asp'])
print("SVM 3-fold subset accuracy " + str(s/3))
plt.show()