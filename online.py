import numpy as np
import pandas as pd
from math import sqrt
from docutils.nodes import header

# 

def sepbyclass(data):
    sep = {}
    for i in range(len(data)):
        vec=data[i]
        if(vec[-1] not in sep):
            sep[vec[-1]]=[]
        sep[vec[-1]].append(vec)
    return sep


def p_x(x,dim, covar, mean):
    
    e = np.linalg.det(covar)
    a = np.transpose(np.subtract(x,mean)) #3x1
    b = np.linalg.inv(covar) #3x3
    c = np.subtract(x,mean) 
    m = np.matmul(b,c)
    m = np.exp(-0.5*np.matmul(a,m))
    
    p= 1/((((2*np.pi)**(dim/2))*sqrt(e))*m)
 
    return p[0][0]
    
    
n_feat=3
n_class=3
n_samples=300

df = pd.read_table('Train.txt',sep='\s+')

# df = df.sample(frac=1).reset_index(drop=True)



array=df.values
sep= sepbyclass(array)
means={}
covars={}
for key in sep:
    sepdata = sep[key]
    sepdata = np.array(sepdata)
#     print(np.shape(sepdata))
    X= sepdata[:,:n_feat]
    Y = sepdata[:, n_feat]
#     print(np.shape(X))
    mean=np.mean(X, axis=0)
    covar = np.cov(X.T)
    if key not in means:
        means[key]=[]
    if key not in covars:
        covars[key]=[]
    means[key].append(mean)
    covars[key].append(covar)
    

tst = pd.read_table('Test.txt',sep='\s+',header=None)
tstdata=tst.values
tst_X=tstdata[:,:n_feat]
tst_Y=tstdata[:,n_feat]

     
# for k in covars.keys():
#     covar = np.reshape(covars[k], (3,3))
#     print(str(k)+': '+str(covar))
# 
# for k in means.keys():
#     mean = np.reshape(means[k], (3,))
#     print(str(k)+': '+str(mean))


# u_X=np. reshape(tst_X[0], (3,1))
# mean = np.reshape(means[1], (3,1))
# covar = np.reshape(covars[1], (3,3))
# p=p_x(u_X, n_feat, covar, mean)
# print(p)

print(len(tst_X))



p_lbls=[]
for i  in  range(len(tst_X)):
    feats=np.reshape(tst_X[i], (3,1))
    print(feats)
    cls=None
    prob=-1
    ar=[]
    for key in means:
        mean = np.reshape(means[key], (3,1))
        covar = np.reshape(covars[key], (3,3))
        p=p_x(feats, n_feat, covar, mean)
        ar.append(p)
        if(p>prob):
            cls=key
            prob=p
    print(ar)
    p_lbls.append(cls)

r_cnt=0 
for i in range(len(p_lbls)):
    print(str(p_lbls[i])+' '+str(tst_Y[i]))
#     if p_lbls[i]==tst_Y[i]:
#         r_cnt=r_cnt+1
print(r_cnt)
    
     
    
    
    