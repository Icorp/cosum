import numpy as np
import matplotlib.pyplot as plt
a = np.random.random((50,50))
print(a)
plt.imshow(a)
plt.colorbar()
plt.show()