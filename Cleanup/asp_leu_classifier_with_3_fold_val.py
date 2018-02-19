import numpy as np
import sys

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
for i in range(3):
    train = [0,1,2]
    train.remove(i)
    val = folds[i]
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

RMSE_error = error / 3
print("RMSE of 3-fold cross validation is " + str(RMSE_error))
print("training data correct percentage " + str(total_correct_train_per/3))
print("test data correct percentage " + str(total_correct_val_per/3))