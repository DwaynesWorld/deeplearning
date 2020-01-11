import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0, 10)
y = x**2
print(x, y)

plt.plot(x, y, 'r--')
plt.xlim(0, 4)
plt.ylim(0, 10)
plt.title("Title")
plt.ylabel("Y Lable")
plt.xlabel("X Label")
plt.show()


mat = np.arange(0, 100).reshape(10, 10)
print(mat)
plt.imshow(mat, cmap='coolwarm')
plt.colorbar()
plt.show()

# Pandas plotting
df = pd.read_csv('../__tensorflow__/00-Crash-Course-Basics/salaries.csv')
print(df)

df.plot(x='Salary', y='Age', kind='scatter')
plt.show()
