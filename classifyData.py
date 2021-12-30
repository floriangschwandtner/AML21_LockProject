# created by Florian Gschwandtner 29.12.2021
# imports
import numpy as np
import matplotlib.pyplot as plt

from LMSQEstimator import LMSQEstimator

# load dataset
data = np.load("Dataset.npy")

input_tensor = data[:, 0:-1, :].transpose(2, 0, 1)
label_tensor = data[:, -1, :].transpose(1, 0)
# print(input_tensor[0,:,:])

# extract information
frame = 0
cutoff_dist = 2.0

condition = ((input_tensor[frame, :, 0]**2 +
             input_tensor[frame, :, 1]**2)**0.5 > cutoff_dist)
# returns [0.725 0.99]
x_vec = np.extract(condition, input_tensor[frame, :, 0])
y_vec = np.extract(condition, input_tensor[frame, :, 1])

# plot


plt.subplot(3, 1, 1)
xbins, xbinvals, _ = plt.hist(x_vec, bins=50)
plt.title('X Histogram')

plt.subplot(3, 1, 2)
ybins, ybinvals, _ = plt.hist(y_vec, bins=50)
plt.title('Y Histogram')


# Median least squares fit
# Get starting values from histogram

# Calc x bin width
bin_width_x = xbinvals[1]-xbinvals[0]

# Find largest bin in x direction
bin_max_x = np.where(xbins == xbins.max())

# Get center of largest bin in x
value_max_x = xbinvals[bin_max_x][0]+0.5*bin_width_x

print('maxbin in x:', value_max_x)

# Calc y bin width
bin_width_y = ybinvals[1]-ybinvals[0]

# Get largest bin in y direction, first half
bin_max_y1 = np.where(ybins[:len(ybins)//2] == ybins[:len(ybins)//2].max())

# Get largest bin in y direction, second half
bin_max_y2 = np.where(ybins[len(ybins)//2:] == ybins[len(ybins)//2:].max())

# Get bin values
value_max_y1 = ybinvals[:len(ybins)//2][bin_max_y1][0]+0.5*bin_width_y
value_max_y2 = ybinvals[len(ybins)//2:][bin_max_y2][0]+0.5*bin_width_y

print('maxbin y first half:', value_max_y1)
print('maxbin y second half:', value_max_y2)

#plt.show()

# Linear LSQ
# Fit two lines in y-direction

# Linear Model y=kx+d
# z = y_vec
# H = [x 1]
boundwidht = 1.0;
x_linspace = np.linspace(x_vec.min(), x_vec.max(), num=20)

condition1 = (y_vec>(value_max_y1-boundwidht)) & (y_vec<(value_max_y1+boundwidht))

H1 = np.extract(condition1, x_vec)
z1 = np.extract(condition1, y_vec)-value_max_y1

#inverse = np.linalg.inv(H2.T@H2)
H_term = H1/(H1.T@H1)
beta1_LSQ = np.matmul(H_term, z1)
y_linear1_LSQ = x_linspace*beta1_LSQ+value_max_y1

beta1 = LMSQEstimator(H1, np.extract(condition1, y_vec))
y_linear1 = x_linspace*beta1[0]+beta1[1]

#LLSQ 2
condition2 = (y_vec>(value_max_y2-boundwidht)) & (y_vec<(value_max_y2+boundwidht))

H2 = np.extract(condition2, x_vec)
z2 = np.extract(condition2, y_vec)-value_max_y2

#inverse = np.linalg.inv(H2.T@H2)
H_term = H2/(H2.T@H2)
beta2_LSQ = np.matmul(H_term, z2)

beta2 = LMSQEstimator(H2, z2)

y_linear2 = x_linspace*beta2[0]+beta2[1]+value_max_y2
y_linear2_LSQ = x_linspace*beta2_LSQ+value_max_y2

# Fit one line in x-direction
boundwidht_x = 3.
condition3 = (x_vec>(value_max_x-boundwidht_x)) & (x_vec<(value_max_x+boundwidht_x))

H3 = np.extract(condition3, x_vec)-value_max_x
z3 = np.extract(condition3, y_vec)

#inverse = np.linalg.inv(H2.T@H2)
H_term = H3/(H3.T@H3)
beta3_LSQ = np.matmul(H_term, z3)

beta3 = LMSQEstimator(H3, z3)

x_linspace2 = np.linspace(H3.min(), H3.max())

y_linear3_LSQ = x_linspace2*beta3_LSQ
y_linear3 = x_linspace2*beta3[0]+beta3[1]

# Show Result
plt.subplot(3,1,3)
plt.plot(x_vec, y_vec, '*')
plt.plot(x_linspace,y_linear1, 'k-')
plt.plot(x_linspace,y_linear1_LSQ, 'r--')

plt.plot(x_linspace,y_linear2, 'k-')
plt.plot(x_linspace,y_linear2_LSQ, 'r--')

#plt.plot(H3, z3, '*')
plt.plot(x_linspace2+value_max_x, y_linear3, 'k-')
plt.plot(x_linspace2+value_max_x, y_linear3_LSQ, 'r--')

ax = plt.gca()

ax.set_xlim([x_vec.min(), x_vec.max()])
ax.set_ylim([y_vec.min(), y_vec.max()])

# Classify points and output a file with points and classification

plt.show()
