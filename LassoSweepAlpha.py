import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
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


start=pow(2, -10)
end=pow(2, 10)
alpha = start
opt_alpha = start
opt_score=10000

all_lower_limit=[]
all_upper_limit=[]
all_alpha=[]
all_avg_acc=[]
while alpha<end:

    model = Lasso(alpha=alpha)

    cv = KFold(
        n_splits=5,
        random_state=42,
        shuffle=True
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

    print("Cross-validation RMSE for α=%f : %f ± %f" % (alpha, -np.mean(scores), np.std(scores)))
    mean=-np.mean(scores)
    std=np.std(scores)
    if  mean < opt_score:
        opt_score= mean
        opt_alpha=alpha
    lower_limit =  mean - std
    upper_limit =  mean + std
    all_avg_acc.append(mean)
    all_lower_limit.append(lower_limit)
    all_upper_limit.append(upper_limit)
    all_alpha.append(alpha)
    alpha*=1.05

plt.plot(all_alpha, all_avg_acc)
plt.fill_between(all_alpha, all_lower_limit, all_upper_limit,color="b",alpha=0.1)
plt.title('average RMSE score with confidence interval')
plt.xlabel('alpha')
plt.ylabel('RMSE score')
plt.legend()
plt.show()
print(opt_alpha)

model = Lasso(alpha=opt_alpha)
print("Fitting linear model over entire training set...", end="")
model.fit(X_train, y_train)
print(" done")

# Compute RMSE
rmse_train = rmse(y_train, model.predict(X_train))
rmse_test = rmse(y_test, model.predict(X_test))

print("Train RMSE = %f, Test RMSE = %f" % (rmse_train, rmse_test))

print("Model parameters:")
print("\t Intercept: %3.5f" % model.intercept_,end="")
for i,val in enumerate(model.coef_):
    if abs(val)>0.001:
        print(", β%d: %3.5f" % (i,val), end="")
print("\n")