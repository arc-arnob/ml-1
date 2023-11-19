import numpy as np
import prtools as pr

import matplotlib.pyplot as plt
import pandas as pd




# %% Ex 1.6 a
# Excercise 1.6 (a)
x = np.array([[ 0.7,0.3,0.2],[2.1,4.5,0]])
df = pd.read_csv("IRIS.csv")
selected_columns_indices = [0, 1, 2]
selected_columns = df.iloc[:, selected_columns_indices]
print(selected_columns.head(10))

x = selected_columns.values
z = x[:5]
y = x[-5:]

x = np.concatenate((z, y))

print(x)

#%% Ex 1.6 b.

print(np.mean(x)) # mean of all values? Mean on flattened Array

print("Axis 0: ", np.mean(x, axis=0)) # Mean alog the column
print("Axis 1: ", np.mean(x, axis=1)) # Mean along the rows





# %% Ex 1.7 a

plt.scatter(x[:,0],x[:,1])

# b. yes

# What above means?
z = [[1,2,3],[11,12,13],[22,23,24]]
z_np_array = np.array(z)
print(z_np_array[:,1])



# %% Ex 1.8 a

# 1 for sesota
# 2 for versicolor
lab = np.array([1,1,1,1,1,2,2,2,2,2]).reshape(10,1)

a = pr.prdataset(x, lab)
print(a)


# %% Ex 1.9

# a
pr.scatterd(a)
pr.scatterd(a, [1,2])



#%% PRE Ex 1.10  theory Matlab way

a = pr.gendatb() # data with labels
b = pr.prdataset(+a) # data without labels
print("Data without Labels: ", b)
print("Data with Labels: ", a)

# boomerangs
boomerangs = pr.boomerangs(10)
print(boomerangs)

pr.scatterd(boomerangs)
pr.scatterd(boomerangs[:, [1,2]])


a = pr.gendatb()
w = pr.nmc(a)
print(w)

b = a*w # Matlab way

print(b)

# %% Python way

a = pr.gendatb()
w = pr.nmc()
w.train(a)
b = w.eval(a)


e = pr.testc(b)



print(e)

# FAILING

pr.scatterd(a)

pr.plotc(w)

# %% Test 2

a = pr.gendath()
print(a)
w = pr.parzenc(a)
pr.scatterd(a)
pr.plotc(w)

# %% Ex 1.13

dataset = 3 + 2.5 * np.random.randn(1000, 2)
print(dataset)

a = pr.prdataset(dataset)
print(a)
pr.scatterd(a)
# pr.plotc(a)


w = pr.gaussm(a)

pr.plotm(w)


#%% Ex 1.15 ERROR
mean1 = 10
mean2= 20
variance = 5

data_points1 = np.random.normal(mean1, np.sqrt(variance), 10)
data_points2 = np.random.normal(mean2, np.sqrt(variance), 10)

# Create labels for each class
labels1 = np.zeros_like(data_points1)  # Class 0
labels2 = np.ones_like(data_points2) # Class 1

data_set = np.concatenate([data_points1, data_points2])
labels = np.concatenate([labels1, labels2])

shuffle_indices = np.arange(len(data_set))
np.random.shuffle(shuffle_indices)

data_set = data_set[shuffle_indices]
labels = labels[shuffle_indices]
print("Data Set: ", data_set)
print("labels: ", labels)

# plt.scatter(data_set, labels)
a = pr.prdataset(data_set, labels) # ERROR

w1 = pr.ldc(a)

w2 = pr.qdc(a)

pr.plotc(w1)

# %% 1.15 redo

# Set the parameters
mean1 = np.array([2, 3])
mean2 = np.array([7, 5])
covariance_matrix = np.array([[2, 1], [1, 2]])

# Generate data points for two 2D normal distributions
data_points1 = np.random.multivariate_normal(mean1, covariance_matrix, 10)
data_points2 = np.random.multivariate_normal(mean2, covariance_matrix, 10)

x = np.concatenate([data_points1, data_points2])

lab = np.array([[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]]).T

labs = np.concatenate([labels1, labels2])

print(x, lab)

a = pr.prdataset(x, lab)
print(a)

w1 = pr.ldc(a)
w2 = pr.qdc(a)
pr.plotc(w1,colors=['red'])

# %%
# Plot the data points
plt.scatter(data_points1[:, 0], data_points1[:, 1], label='Distribution 1')
plt.scatter(data_points2[:, 0], data_points2[:, 1], label='Distribution 2')


# Plot the means
plt.scatter(mean1[0], mean1[1], color='red', marker='x', label='Mean 1')
plt.scatter(mean2[0], mean2[1], color='blue', marker='x', label='Mean 2')

# Add axis labels and a legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Add a title
plt.title('Two 2D Normal Distributions with Different Means')

# Show the plot
plt.show()

# %% Ex 1.16
# Set the number of samples for each class
num_samples = 10

# Generate data points for the first class from a uniform distribution
class1_uniform = np.random.rand(num_samples, 2)  # 2D data

# Generate data points for the second class from a normal distribution
class2_normal = np.random.randn(num_samples, 2)   # 2D data

# Combine data points for both classes
data_set_2d = np.vstack((class1_uniform, class2_normal))
labels = np.concatenate([np.zeros(num_samples), np.ones(num_samples)])

a = pr.prdataset(data_set_2d, labels)

print(a)

t1 = pr.ldc(a)
t2 = pr.qdc(a)

pr.scatterd(a)

pr.plotc(t1, colors=['red'])
pr.plotc(t2, colors=['blue'])


# %% Ex 1.17 Density Estimation
a = pr.gendats([20,20],1,8)


h = 1
a = pr.prdataset(+a)
w = pr.parzenm(a,h)

pr.scatterd(a); pr.plotm(w, colors=['red'])



pr.plotm(w,gridsize=100, colors=['blue'])


# %% Ex 1.18
a = pr.gendats([20,20],1,8) # Generate data
a = pr.prdataset(+a)
hs = [0.01,0.05,0.1,0.25,0.5,1,1.5,2,3,4,5] # Array of h’s to try
LL = np.zeros(len(hs))
for i in range(len(hs)): # For each h...
    w = pr.parzenm(a,hs[i]) # estimate Parzen density on training set?
    LL[i] = np.sum(np.log(+(a*w))); # calculate log-likelihood
plt.plot(hs,LL); 



# %% Ex 1.19

[trn,tst] = pr.gendat(a,0.5)
hs = [0.01,0.05,0.1,0.25,0.5,1,1.5,2,3,4,5]
Ltrn = np.zeros(len(hs))
Ltst = np.zeros(len(hs))
for i in range(len(hs)): # For each h...
    w = pr.parzenm(trn,hs[i]) # estimate Parzen density on training set
    Ltrn[i] = np.sum(np.log(+(trn*w))) # calculate trn log-likelihood
    Ltst[i] = np.sum(np.log(+(tst*w))) # calculate tst log-likelihood

plt.plot(hs,Ltrn,’b-’) # Plot trn log-likelihood as function of h
plt.plot(hs,Ltst,’r-’) # Plot tst log-likelihood as function of h





