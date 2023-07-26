
import numpy as np
import matplotlib.pyplot as plt
import torch

class BS_curve(object):
    def __init__(self,n,p,cp=False,knots=None):
        self.n = n
        self.p = p
        if cp:
            self.cp = cp
            self.u = knots
            self.m = knots.shape[0]-1
        else:
            self.cp = None
            self.u = None
            self.m = None
        self.paras = None

    def check(self):
        if self.m == self.n + self.p + 1:
            return 1
        else:
            return 0

    def coeffs(self,uq):
       
       
        # algorithm is from https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-curve-coef.html
    
       
       
       
        N = np.zeros(self.m+1,dtype=np.float64) 

       
        if uq == self.u[0]:
            N[0] = 1.0
            return N[0:self.n+1]
        elif uq == self.u[self.m]:
            N[self.n] = 1.0
            return N[0:self.n+1]

       
       
        check = uq - self.u
        ind = check >=0
        k = np.max(np.nonzero(ind))
       
        sk = np.sum(self.u==self.u[k])
        N[k] = 1.0
       
        for d in range(1,self.p+1):
            r_max = self.m - d - 1
            if k-d >=0:
                if self.u[k+1]-self.u[k-d+1]:
                    N[k-d] = (self.u[k+1]-uq)/(self.u[k+1]-self.u[k-d+1])*N[k-d+1]
                else:
                    N[k-d] = (self.u[k+1]-uq)/1*N[k-d+1]

            for i in range(k-d+1,(k-1)+1):
                if i>=0 and i<=r_max:
                    Denominator1 = self.u[i+d]-self.u[i]
                    Denominator2 = self.u[i+d+1]-self.u[i+1]
                   
                    if Denominator1 == 0:
                        Denominator1 = 1
                    if Denominator2 == 0:
                        Denominator2 = 1
                    N[i] = (uq-self.u[i])/(Denominator1)*N[i]+(self.u[i+d+1]-uq)/(Denominator2)*N[i+1]

            if k <= r_max:
                if self.u[k+d]-self.u[k]:
                    N[k] = (uq-self.u[k])/(self.u[k+d]-self.u[k])*N[k]
                else:
                    N[k] = (uq-self.u[k])/1*N[k]
        return N[0:self.n+1]


    def De_Boor(self,uq):
       
       
       
        check = uq - self.u
        ind = check >=0
        k = np.max(np.nonzero(ind))
        
       
        if uq in self.u:
           
            sk = np.sum(self.u==self.u[k])
            h = self.p - sk
        else:
            sk = 0
            h = self.p

       
        if h == -1:
            if k == self.p:
                return np.array(self.cp[0])
            elif k == self.m:
                return np.array(self.cp[-1])

        # initial values of P(affected control points) >>> Pk-s,0 Pk-s-1,0 ... Pk-p+1,0
        P = self.cp[k-self.p:k-sk+1]
        P = P.copy()
        dis = k-self.p
       
        for r in range(1,h+1):
           
            temp = []
            for i in range(k-self.p+r,k-sk+1):
                a_ir = (uq-self.u[i])/(self.u[i+self.p-r+1]-self.u[i])
                temp.append((1-a_ir)*P[i-dis-1]+a_ir*P[i-dis])
            P[k-self.p+r-dis:k-sk+1-dis] = np.array(temp)
       
       
        return P[-1]

    def bs(self,us):
        y = []
        for x in us:
            y.append(self.De_Boor(x))
        y = np.array(y)
        return y

    def estimate_parameters(self,data_points,method="centripetal"):
        pts = data_points.copy()
        N = pts.shape[0]
        w = pts.shape[1]
        Li = []
        for i in range(1,N):
            Li.append(np.sum([pts[i,j]**2 for j in range(w)])**0.5)
        L = np.sum(Li)

        t= [0]
        for i in range(len(Li)):
            Lki = 0
            for j in range(i+1):
                Lki += Li[j]
            t.append(Lki/L)
        t = np.array(t)
        self.paras = t
        ind = t>1.0
        t[ind] = 1.0
        return t


    def get_knots(self,method="average"):
        knots = np.zeros(self.p+1).tolist()
        paras_temp = self.paras.copy()
       
        self.m = self.n + self.p + 1
       
       
        num = self.m - self.p 

        ind = np.linspace(0,paras_temp.shape[0]-1,num)
        ind = ind.astype(int)
        paras_knots = paras_temp[ind]

        for j in range(1,self.n-self.p+1):
            k_temp = 0
           
            for i in range(j,j+self.p-1+1):
                k_temp += paras_knots[i]
            k_temp /= self.p
            knots.append(k_temp)

        add = np.ones(self.p+1).tolist()
        knots = knots + add
        knots = np.array(knots)
        self.u = knots
        self.m = knots.shape[0]-1
        return knots

    def set_paras(self,parameters):
        self.paras = parameters

    def set_knots(self,knots):
        self.u = knots

    def approximation(self,pts):
        ## Obtain a set of parameters t0, ..., tn
       
       
       
        num = pts.shape[0]-1
        P = np.zeros((self.n+1,pts.shape[1]),dtype=np.float64)
        P[0] = pts[0]
        P[-1] = pts[-1]
       
        N = []
        for uq in self.paras:
            N_temp = self.coeffs(uq)
            N.append(N_temp)
        N = np.array(N)

        Q = [0]
        for k in range(1,num-1+1):
            Q_temp = pts[k] - N[k,0]*pts[0] - N[k,self.n]*pts[-1]
            Q.append(Q_temp)

        b = [0]
        for i in range(1,self.n-1+1):
            b_temp = 0
            for k in range(1,num-1+1):
                b_temp += N[k,i]*Q[k]
            b.append(b_temp)
       
        b = b[1::]
        b = np.array(b)
       
       
        #     b=np.hstack((b.reshape(b_shape,1),b.reshape(b_shape,1)))
       
        N = N[:,1:(self.n-1)+1]
        A = np.dot(N.T,N)
        #print('shape',A.shape,b.shape)
        #cpm = np.linalg.solve(A,b)
        cpm=np.matmul(np.linalg.pinv(A),b)
        #print(cpm.shape,A.shape,b.shape)
       
        #print('----',np.linalg.pinv(A).shape,b.shape,cpm.shape,P[1:self.n].shape)
        
        P[1:self.n] = cpm
        self.cp = P
        return P


def b_spline_basis(i, k, u, nodeVector):
   
   
    if k == 0:
        if (nodeVector[i] < u) & (u <= nodeVector[i + 1]): 
            result = 1
        else:
            result = 0
    else:
       
        length1 = nodeVector[i + k] - nodeVector[i]
        length2 = nodeVector[i + k + 1] - nodeVector[i + 1]
       
        if length1 == 0: 
            alpha = 0
        else:
            alpha = (u - nodeVector[i]) / length1
        if length2 == 0:
            beta = 0
        else:
            beta = (nodeVector[i + k + 1] - u) / length2
   
        result = alpha * b_spline_basis(i, k - 1, u, nodeVector) + beta * b_spline_basis(i + 1, k - 1, u, nodeVector)
    return result

# def cal_matrix_bspline(n,k,nodeVector,X,Y):
#     dim=100
#     basis_matrix=torch.zeros((n,dim))
#     rx = torch.zeros(dim) 
#     ry = torch.zeros(dim)
#     for i in range(n): 
#         U = torch.linspace(nodeVector[k], nodeVector[n], dim) 
#         j = 0
#         for u in U:
#            
#             basis_matrix[i,j]=b_spline_basis(i, k, u, nodeVector)
#             j = j + 1
#        
#        
#         print('------',rx.shape,basis_matrix.shape)
#         rx = rx + X[i] * basis_matrix[i,:]
#         ry = ry + Y[i] * basis_matrix[i,:]
#        
#        
#        
#         rx[0],ry[0]=X[0],Y[0]
#     return rx,ry

# def cal_matrix_bspline(n,k,nodeVector,X,Y,dim=100):
#    
#     device_xy=X.device
#     basis_matrix=torch.zeros((n,dim))
#     rx = torch.zeros(dim).to(device_xy) 
#     ry = torch.zeros(dim).to(device_xy)
#     for i in range(n): 
#         U = torch.linspace(nodeVector[k], nodeVector[n], dim) 
#         j = 0
#         for u in U:
#            
#             basis_matrix[i,j]=b_spline_basis(i, k, u, nodeVector)
#             j = j + 1
#        
#        
#         #print('----',rx.shape,X[i].shape,basis_matrix[i,:].shape)
#         rx = rx + X[i] * basis_matrix[i,:].to(device_xy)
#         ry = ry + Y[i] * basis_matrix[i,:].to(device_xy)
#        
#        
#        
#    
#    
#     rx[0],ry[0]=X[0],Y[0]
#    
#    
#    
#     return rx,ry

def cal_matrix_bspline(n,k,nodeVector,X,Y,dim=100):
   
    device_xy=X.device
    basis_matrix=torch.zeros((n,dim))
    rx = torch.zeros(dim).to(device_xy) 
    ry = torch.zeros(dim).to(device_xy)
    U = torch.linspace(nodeVector[k], nodeVector[n], dim) 
    for i in range(n): 
       
       
       
        basis_matrix[i]=torch.tensor([b_spline_basis(i, k, U[j], nodeVector) for j in range(dim)])
       
       
        #print('----',rx.shape,X[i].shape,basis_matrix[i,:].shape)
        rx = rx + X[i] * basis_matrix[i,:].to(device_xy)
        ry = ry + Y[i] * basis_matrix[i,:].to(device_xy)

       
       
       
   
   
    rx[0],ry[0]=X[0],Y[0]
   
   
   
    return rx,ry

# def cal_matrix_all(n,k,nodeVector,X,Y,dim=100):
#     matrix_all=torch.zeros((X.shape[0],dim,2))
#     for num in range(X.shape[0]):
#         x_temp,y_temp=cal_matrix_bspline(n,k,nodeVector,X[num,:],Y[num,:],dim)
#         matrix_temp=torch.hstack((x_temp.reshape(x_temp.shape[0],1),y_temp.reshape(y_temp.shape[0],1)))
#         matrix_all[num]=matrix_temp
#     return matrix_all

def cal_matrix_all(n,k,nodeVector,X,Y,dim=100):
    matrix_all=torch.zeros((X.shape[0],dim,2))
   
   
    #     matrix_temp=torch.hstack((x_temp.reshape(dim,1),y_temp.reshape(dim,1)))
   
    matrix_temp=[torch.hstack((cal_matrix_bspline(n,k,nodeVector,X[num,:],Y[num,:],dim)[0].reshape(dim,1),cal_matrix_bspline(n,k,nodeVector,X[num,:],Y[num,:],dim)[1].reshape(dim,1))) for num in range(X.shape[0])]
   
    for num in range(X.shape[0]):
        matrix_all[num]=matrix_temp[num]
   
    return matrix_all