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
randint_wseed = np.random.randint(0, 100, 10)
print(randint_wseed)
