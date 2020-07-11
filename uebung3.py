import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

iris = np.loadtxt("iris.txt", delimiter = ",")

def PCA(normalized_data):
    cov = np.cov(normalized_data)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    summe = np.sum(eigenvalues)
    eigenvalues_prop = eigenvalues/summe
    
    ind = [1,2,3,4]
    kum = np.zeros(4)
    kum[0] = eigenvalues_prop[0]
    for i in range(1,4):
        kum[i]=kum[i-1] + eigenvalues_prop[i]  
    kum=kum*100
    plt.bar(ind, kum)
    rounded=np.round(kum,2)
    for idx,y in enumerate(rounded):
        plt.text(ind[idx]-0.2,y+1,str(y)+"%")
    plt.show()    
    return eigenvalues, eigenvectors
    
def threec(eigenvalues, eigenvectors, normalized_data, pred):
    B = eigenvectors[:,0:2]
    normalized_data_p = np.matmul(B.T, normalized_data)#*(-1) # to get equal things as falko
    normalized_data_p = np.vstack((normalized_data_p, pred))
    colors = ['red','green','blue']
    plt.scatter(normalized_data_p[0], normalized_data_p[1], c=normalized_data_p[2], cmap=matplotlib.colors.ListedColormap(colors))
    return

def rmse(x):
    return np.sqrt(np.mean(x**2))

def comp_set(n, eigenvectors, normalized_data, mean, var):
    B = eigenvectors[:,0:n+1]
    an = np.matmul(B.T, normalized_data) 
    reconstruction = np.matmul(B, an)
    rec=np.multiply(reconstruction.T,var)
    rec=rec+mean
    return rec
        
def threed(eigenvalues, eigenvectors, normalized_data, pre_iris, mean, var):
    lsg=np.zeros([4,4])
    for comp in range(4):
        re = comp_set(comp, eigenvectors, normalized_data, mean, var)
        diff = re - pre_iris
        for feature in range(4):
            print(100*np.sqrt(mean_squared_error(re[:,feature],pre_iris[:,feature]))/(np.mean(pre_iris[:,feature])))
            lsg[comp,feature]=rmse(diff[:,feature])/(np.sum(pre_iris[:,feature]/len(pre_iris)))
    print("lsg:", lsg*100)
    return 

def zca_whitening(epsilon):
    pre_iris = iris[:,0:4]
    mean = pre_iris.mean(0)
    xstern = (pre_iris - mean).T
    cov = np.cov(xstern)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    xPCAwhite=np.diag(1./np.sqrt(eigenvalues+epsilon))@eigenvectors.T@xstern
    xZCAwhite=eigenvectors@xPCAwhite
    return xZCAwhite
def zca_brunner(x,epsilon):
    if x.shape[0]>x.shape[1]:
        raise
    evals,evecs=np.linalg.eigh(np.cov(x))
    evals=evals+epsilon
    z = evecs @ np.diag(evals**(-1/2)) @ evecs.T @ x
    return z
    
if __name__ == "__main__":
    #3a
    n = len(iris)
    pre_iris = iris[:,0:4]
    pred = iris[:,4]
    mean = pre_iris.mean(0)
    step1 = pre_iris - mean
    standard_deviation=np.std(pre_iris, axis=0)
    normalized_data = np.multiply(step1, 1/standard_deviation).T   
    print(normalized_data.shape)
    #3b
    eigenvalues, eigenvectors = PCA(normalized_data)
    
    #3c
    # we use components 1 and 2
    threec(eigenvalues, eigenvectors, normalized_data, pred)
    
    #3d
    threed(eigenvalues, eigenvectors, normalized_data, pre_iris, mean, standard_deviation)
    #de
    epsilon=1*10**(-5)
    print(epsilon)
    print(zca_brunner(pre_iris.T,epsilon)[2,2])
    print(zca_whitening(epsilon)[2,2])