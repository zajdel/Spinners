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
                          title='Asp Concentration',
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
    plt.clim((0,30))
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

path120 = "feature_csvs_120s/" # path to feature csv's
path90 = "feature_csvs_90s/" 
path60 = "feature_csvs_60s/" 
path30 = "feature_csvs_30s/"
path15 = "feature_csvs_15s/"

count = int(sys.argv[1])

asp100n_15 = np.loadtxt(path15 + "100n_asp.csv", delimiter=",")
asp100n_30 = np.loadtxt(path30 + "100n_asp.csv", delimiter=",")
asp100n_60 = np.loadtxt(path60 + "100n_asp.csv", delimiter=",")
asp100n_90 = np.loadtxt(path90 + "100n_asp.csv", delimiter=",")
asp100n_120 = np.loadtxt(path120 + "100n_asp.csv", delimiter=",")

asp1u_15 = np.loadtxt(path15 + "1u_asp.csv", delimiter=",")
asp1u_30 = np.loadtxt(path30 + "1u_asp.csv", delimiter=",")
asp1u_60 = np.loadtxt(path60 + "1u_asp.csv", delimiter=",")
asp1u_90 = np.loadtxt(path90 + "1u_asp.csv", delimiter=",")
asp1u_120 = np.loadtxt(path120 + "1u_asp.csv", delimiter=",")

aspcon_15 = np.loadtxt(path15 + "control_asp.csv", delimiter=",")
aspcon_30 = np.loadtxt(path30 + "control_asp.csv", delimiter=",")
aspcon_60 = np.loadtxt(path60 + "control_asp.csv", delimiter=",")
aspcon_90 = np.loadtxt(path90 + "control_asp.csv", delimiter=",")
aspcon_120 = np.loadtxt(path120 + "control_asp.csv", delimiter=",")

#shuffle
np.random.shuffle(asp100n_15[:,0])
np.random.shuffle(asp100n_30[:,0])
np.random.shuffle(asp100n_60[:,0])
np.random.shuffle(asp100n_90[:,0])
np.random.shuffle(asp100n_120[:,0])
np.random.shuffle(asp1u_15[:,0])
np.random.shuffle(asp1u_30[:,0])
np.random.shuffle(asp1u_60[:,0])
np.random.shuffle(asp1u_90[:,0])
np.random.shuffle(asp1u_120[:,0])
np.random.shuffle(aspcon_15[:,0])
np.random.shuffle(aspcon_30[:,0])
np.random.shuffle(aspcon_60[:,0])
np.random.shuffle(aspcon_90[:,0])
np.random.shuffle(aspcon_120[:,0])

# biases, cw, ccw, switch
asp100n = iter(zip(asp100n_15[:,0], asp100n_30[:,0], asp100n_60[:,0],  asp100n_90[:,0], asp100n_120[:,0]))
asp1u = iter(zip(asp1u_15[:,0],asp1u_30[:,0], asp1u_60[:,0], asp1u_90[:,0], asp1u_120[:,0]))
aspcon = iter(zip(aspcon_15[:,0],aspcon_30[:,0], aspcon_60[:,0], aspcon_90[:,0], aspcon_120[:,0]))

training_set = np.zeros((count, 5))

counter = 0
prev_count = counter
while counter < count:
    try:
        a, b, c, d, e = next(aspcon)
        training_set[counter] = np.asarray([ b,c,d, e, -1])
        counter += 1
    except StopIteration:
        pass
    if counter < count:
        try:
            f,g,h,i,j = next(asp100n)
            training_set[counter] = np.asarray([g,h,i, j,0])
            counter += 1
        except StopIteration:
            pass
    if counter < count:
        try:
            k,l,m,n,o = next(asp1u)
            training_set[counter] = np.asarray([l,m,n,o, 1])
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

folds = {0:training_set[0:count/3],1:training_set[count/3:2*count/3],2:training_set[2*count/3:count]}
error = 0
total_correct_train_per = 0
total_correct_val_per = 0
s = 0.0
hl = 0.0
cnf_matrix=np.zeros((3,3),np.int8)
for i in range(3):
    train = [0,1,2]
    train.remove(i)
    val = folds[i]
    t = np.concatenate((folds[train[0]],folds[train[1]]),axis=0)
    X = t[:,0:3]
    Y = t[:,-1]
    X_val = val[:,0:3]
    Y_val = val[:,-1]
    
    clf = svm.SVC(kernel='poly')
    clf.fit(X,Y)
    s += clf.score(X_val, Y_val)
    Y_pred = clf.predict(X_val)
    cnf_matrix+=confusion_matrix(Y_val,Y_pred)
    print clf
	
print("SVM 3-fold subset accuracy " + str(s/3))
plot_confusion_matrix(cnf_matrix,classes=['0 M','100 nM',r'1 $\mu$M'])
plt.show()
