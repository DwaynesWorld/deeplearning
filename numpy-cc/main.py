import numpy as np

my_list = [1, 2, 3]
print(my_list)
print(type(my_list))

my_array = np.array(my_list)
print(my_array)
print(type(my_array))

range = np.arange(0, 10, 2)
print(range)

zeros = np.zeros((2, 5))
print(zeros)

ones = np.ones(5)
print(ones)

linspace = np.linspace(0, 11, 5)
print(linspace)

randint = np.random.randint(5, 5000, (3, 3))
print(randint)

np.random.seed(101)
arr = np.random.randint(0, 100, 10)
print(arr)
print(arr.max())
print(arr.argmax())
print(arr.reshape(2, 5))

mat = np.arange(0, 100).reshape(10, 10)
print(mat)
print(mat[4, 1])
print(mat[:, 0])
print(mat[5])
print(mat[0:3, 0:3])
filter = mat > 50
print(filter)
print(mat[filter])
print(mat[mat > 55])
