import json
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np



fitness = []
t_total = []
with open('results/data.json') as json_file:
    data = json.load(json_file)
    for p in range(len(data)):
        for t in data['pop'+str(p)]:
            fitness.append(t['fitness'])

for i in range(200000):
    t_total.append(i)

# create some y data points
ydata1 = np.array(t_total)
xdata1 = np.array(fitness)

print(ydata1)
# sort the data so it makes clean curves
xdata1.sort()



# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xdata1, ydata1, color='tab:blue')


plt.stem(xdata1, ydata1)
plt.show()