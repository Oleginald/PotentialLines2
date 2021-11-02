import sys
a = 2.
b = True
x1 = sys.getsizeof(a)
x2 = sys.getsizeof(b)
print(False.__sizeof__())
print(True.__sizeof__())
print(int.__sizeof__(1))
print(float.__sizeof__(1.654465))
