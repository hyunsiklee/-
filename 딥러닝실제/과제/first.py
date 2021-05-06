# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

hungry=True

sleepy=False

hungry and sleepy



if hungry:
    print("i'm hungry")
else :
    print("i'm not hungry")
    

for i in[1,2,3]:
    print(i)


def hello():
    print("hello World")
    


def hello(object):
    print("hello " + object)
    



hello("cat")


class Man:
    def __init__(self, name):
        self.name = name
        print("Inint Done")
        
    def hello(self):
        print("hello " + self.name)
        
    def goodbye(self):
        print("goodbye " + self.name)
        
m= Man('david')
m.hello()
m.goodbye()

import numpy as np

x = np.array([1.0,2.0,3.0])
print(x)

type(x)

x= np.array([1.0,2.0,3.0])
y= np.array([2.0,4.0,6.0])
print(x+y)
print(x-y)
print(x*y)
print(x/y)


print(x/2)

A = np.array([[1,2],[3,4]])
print(A)

print(A.shape) #몇바이볓이냐

print(A.dtype) # 왜 32 비트냐

B=np.array([[3,0],[0,6]])

print(A+B)
print(A*B)


A = np.array([[1,2],[3,4]])
B = np.array([10,20])
print(A*B)

X = np.array([[51,55],[14,19],[0,4]])
print(X)
print(X[0])
print(X[0][1])

print("----------------")
for row in X:
    print(row)



