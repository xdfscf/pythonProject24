import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse

from data_generator import postfix
def lift(x):
    new_x= x.tolist()
    d=len(x)

    for i in range(d):
        for j in range(i+1):
            new_x.append(new_x[i]*new_x[j])

    return new_x

def liftDataset(data):
    data=[lift(x) for x in data]
    return data
# Number of samples
N = 1000

# Noise variance
sigma = 0.01

# Feature dimension
d = 40

psfx = postfix(N, d, sigma)

X = np.load("X" + psfx + ".npy")
y = np.load("y" + psfx + ".npy")
X=liftDataset(X)
X=np.array(X)
print("Dataset has n=%d samples, each with d=%d features," % X.shape, "as well as %d labels." % y.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0], X_test.shape[0]))


x=[]
y1=[]
y2=[]
for fr in range(1,11):
    fraction=fr*0.1
    if fraction<0.95:
        X_train_fr, _, y_train_fr, _ = train_test_split(
            X_train, y_train, test_size=1-fraction, random_state=42)
    else:
        X_train_fr = X_train
        y_train_fr = y_train
    print("using %f fraction of training set for fitting" % (fraction))
    print("Fitting linear model...", end="")
    model = LinearRegression()
    model.fit(X_train_fr, y_train_fr)
    print(" done")

    # Compute RMSE on train and test sets
    rmse_train = rmse(y_train_fr, model.predict(X_train_fr))
    rmse_test = rmse(y_test, model.predict(X_test))

    print("Train RMSE = %f, Test RMSE = %f" % (rmse_train, rmse_test))

    print("Model parameters:")
    print("\t Intercept: %3.5f" % model.intercept_, end="")
    for i, val in enumerate(model.coef_):
        print(", β%d: %3.5f" % (i, val), end="")
    print("\n")

    x.append(fr)
    y1.append(rmse_train)
    y2.append(rmse_test)


fig, ax = plt.subplots()

ax.plot(x, y1, label='In sample MSE')
ax.plot(x, y2, label='out of sample MSE')

ax.set_xlabel('fraction')
ax.set_ylabel('MSE')
ax.set_title('MSE')
ax.legend()

plt.show()
