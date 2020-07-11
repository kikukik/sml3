# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:41:36 2020

@author: No-Pa
"""


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

iris= np.loadtxt("iris.txt", delimiter=",")
n=len(iris)
means=np.mean(iris,axis=0)
data=iris[:,:-1]-means[:-1]
var=np.std(data, axis=0)
normalized_data=data/var
normalized_data_l=np.c_[normalized_data, iris[:,4]]
print(np.shape(normalized_data))
print(np.shape(normalized_data_l))

cov=np.cov(normalized_data.T)
eigvalues,eigvectors=np.linalg.eig(cov)
print ("eigvals:", eigvalues)
print("eigenvectors(spalten):",eigvectors)
e1=eigvectors[:,0]
print("utCU:",e1.T@cov@e1)
def sk():   
    pca=PCA(n_components=4)
    pca.fit(iris[:,:-1])
    evr=pca.explained_variance_ratio_
    print(evr)
    return 

def pca_selbst():
    s=sum(eigvalues)
    eigval_perc=eigvalues/s
    print(eigval_perc)
    s1=eigval_perc[0]
    s2=sum(eigval_perc[:2])
    s3=sum(eigval_perc[:3])
    s4=sum(eigval_perc[:4])
    print(s1,s2,s3,s4)
    plt.bar([1,2,3,4],[s1,s2,s3,s4])
    v12=eigvectors[:,0:2]
    new_data=np.matmul(v12.T,normalized_data.T)
    plt.figure()
    new_data=np.vstack((new_data,data[:,3]))
    colors=['red','g','b']
    plt.scatter(new_data[0,:],new_data[1,:],c=new_data[2,:],cmap=matplotlib.colors.ListedColormap(colors))
    return

def rmse(x):
    return np.sqrt(np.mean(x**2))

def get_xn_schlange(mean,points,var,vecs,n):
    points=np.add(points,-np.mean(points,axis=0))
    print("lv:",len(vecs))
    for i in range(n):
        u_i=vecs[:,i]
        a_n=np.matmul(u_i.T,points.T)
        print("anshape:",np.shape(a_n))
        np.add(mean,np.matmul(u_i,a_n))       
    return

def reconstruct(X, mean, var):
    X_step1 = np.multiply(X.T, var)
    X_step2 = X_step1 + mean
    return X_step2

def comp_set(n, eigenvectors, normalized_data, mean, var):
    B = eigenvectors[:,0:n+1]
 #   print("B: ",B)
    normalized_data_p = np.matmul(B.T, normalized_data) 
   # print("n_data:", normalized_data_p)
    reconstruction = np.matmul(B, normalized_data_p)
#    print("recons:",np.shape(reconstruction))
    reconstruct1 = reconstruct(reconstruction, mean, var)
 #   print("recons1:",np.shape(reconstruct1))
    return reconstruct1



def get_nrmse(points,mean,var):
    print("p:",points[55,2])
    nrmse=np.zeros([4,4])
    for j in range(4):
        vecs=eigvectors[:,0:j+1]
        # get_xn_schlange(mean,points,var,vecs,j)              
        # print(np.shape(vecs.T),np.shape(normalized_data))
        # a_n=np.matmul(normalized_data,vecs).T
        # x_schlange_n=np.matmul(vecs,a_n).T
        x_schlange_n=comp_set(j, eigvectors, normalized_data.T, mean, var)
        print(np.shape(x_schlange_n),np.shape(points))
        print("xs:",x_schlange_n[1,1])
        diff=x_schlange_n-points
        print("shapeD:",np.shape(diff))
        print("mein_m:", np.mean(points[j,:]))
        for i in range(4):
            nrmse[i,j]=np.sqrt(mean_squared_error(x_schlange_n[:,i],points[:,i]))/(np.mean(points[:,i]))
    return nrmse.T

print("nrmse:", get_nrmse(iris[:,0:4],means[:-1],var))
sk()
pca_selbst()

    
