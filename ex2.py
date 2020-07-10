# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np
from matplotlib import pyplot as plt

data= np.loadtxt("ldaData.txt")

C_1=data[0:50]
C_2=data[50:93]
C_3=data[93:137]

plt.scatter(C_1[:,0],C_1[:,1])
plt.scatter(C_2[:,0],C_2[:,1])
plt.scatter(C_3[:,0],C_3[:,1])
plt.figure()
def get_var(mean,points):
    l=points-mean
    s=0
    for wer in l:
        s=s+wer**2
    s=s/(len(l)-1)
    return s
def gaussian(x,mu,sigma):
    # sigma=sigma**2
    y=np.exp(-((x-mu)**2)/(2*sigma))/((2*np.pi*sigma)**0.5)
    return y

def get_w(ClassA, ClassB):
    n_1 = len(ClassA)
    n_2 = len(ClassB)
    m_1 = np.mean(ClassA, axis=0)
    m_2 = np.mean(ClassB, axis=0)

    sum_1 = np.zeros(2)
    for i in range(n_1):
        sum_1 = sum_1 + np.outer(ClassA[i] - m_1,(ClassA[i] - m_1))
    sum_2 = np.zeros(2)
    for i in range(n_2):
        sum_2 = sum_2 + np.outer(ClassB[i] - m_2, (ClassB[i] - m_2))
    Sw=sum_1+sum_2
    w=np.matmul(np.linalg.inv(Sw),m_1-m_2)
    return w

def compare_classes(ClassA,iA,ClassB,iB,state):
    w=get_w(ClassA,ClassB)
    x1=np.matmul(ClassA,w)
    x2=np.matmul(ClassB,w)
    
    prior1=len(x1)/(len(x1)+len(x2))
    prior2=len(x2)/(len(x1)+len(x2))
    m1=np.sum(x1)/len(x1)
    print(m1)
    m2=np.sum(x2)/len(x2)
    print(m2)
    v1=get_var(m1,x1)
    v2=get_var(m2,x2)
    print(v1)
    xg=np.matmul(data,w)
    for index,x in enumerate(xg):
        g1=gaussian(x,m1,v1)
        g2=gaussian(x,m2,v2)
        if (g1/g2)>(prior2/prior1):
          #  print("decide class 1")
            state[iA-1,index]=state[iA-1,index]+1
        else:
           # print("decide class 2")
            state[iB-1,index]=state[iB-1,index]+1
    return state

def plotting():
    state=np.zeros([3,len(data)])
    compare_classes(C_1,1,C_2,2,state)
    compare_classes(C_2,2,C_3,3,state)
    compare_classes(C_1,1,C_3,3,state)
    print(state)
    klassen=np.argmax(state,axis=0)
    for index,d in enumerate(data):
        if klassen[index]==0:
            plt.scatter(data[index,0],data[index,1],c='red')
        elif klassen[index]==1:
            plt.scatter(data[index,0],data[index,1],c='yellow')
        elif klassen[index]==2:
            plt.scatter(data[index,0],data[index,1],c='black')
    s=0
    
    for idx in range(137):
        if idx<50:
            if klassen[idx]!=0:
                s=s+1
        else:
            if idx<93:
                if klassen[idx]!=1:
                    s=s+1
            else:
                if klassen[idx]!=2:
                    s=s+1
    print(s)
    return 

plotting()

        
    

            
    









