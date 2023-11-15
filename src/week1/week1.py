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


