import numpy as np

# Generate 1000 random points
x = np.zeros((3,3))
x = np.random.random((3,3,3))
print x
print x[0,0,1]
print x.reshape((-1,1,1))
# Plot them
