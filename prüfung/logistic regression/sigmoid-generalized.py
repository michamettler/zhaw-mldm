import numpy as np
'''EXAMPLE'''
'''
    wheigts: 0 = 14, 1 = -0.00035, 2 = -1.2
    z: 1 = 35000, 2 = 2
'''

'''INPUT'''
b = -12
w = np.array([0.0002,-0.00001])
z = np.array([100_000,500_000])

'''INPUT'''

result = 1 / (1 + np.exp(-(b + sum(w*z))))
print(f'{result:.3f}, in prozent: {result*100:.3f}%')