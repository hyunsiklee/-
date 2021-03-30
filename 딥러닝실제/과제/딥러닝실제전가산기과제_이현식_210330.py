#전가산기 과제

import numpy as np

def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1

def NAND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1    
    
def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0
    else :
        return 1   
    
def XOR(x1, x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    return AND(s1, s2)


def 전가산기(x,y,c):
    a1 = XOR(x,y)
    b1 = AND(x,y)
    s = XOR(a1,c)
    a2 = AND(a1,c)
    c2 = OR(a2,b1)
    result = np.array([s,c2])
    return result

print('[x y z] = [s c]')
print('[0 0 0] =',전가산기(0,0,0))
print('[0 0 1] =',전가산기(0,0,1))
print('[0 1 0] =',전가산기(0,1,0))
print('[0 1 1] =',전가산기(0,1,1))
print('[1 0 0] =',전가산기(1,0,0))
print('[1 0 1] =',전가산기(1,0,1))
print('[1 1 0] =',전가산기(1,1,0))
print('[1 1 1] =',전가산기(1,1,1))


