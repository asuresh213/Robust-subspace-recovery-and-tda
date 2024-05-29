from scipy.linalg import null_space
from scipy.linalg import orth
import numpy as np
import random

np.random.seed(1200)
random.seed(1)


class RSR:
    def __init__(self, data, epsilon):
        self.data = data
        self.epsilon = epsilon 
        self.m = len(data)
        self.n = len(data[0])
        self.d = [len(i[0]) for i in data]
        self.D = np.sum(self.d)
        self.N = self.n*self.D
        self.epsilon_upper = int((2*self.N**2)/epsilon)
        self.S = np.random.randint(2, size = self.epsilon_upper)
        self.tol = 1e-8

    def build_B_inv_matrix(self):
        B = np.zeros((self.N,self.N))
        i=0
        prev_d = len(self.data[0][0])
        for Xi in self.data:
            cur_d = len(Xi[0])
            j=0
            for vij in Xi:
                for q in range(self.n):
                    for r in range(self.D):
                        s = random.choice(self.S) 
                        to_enter = s*vij
                        B[q*cur_d + i*self.n*prev_d:(q+1)*cur_d + i*self.n*prev_d, r+j*self.D] = to_enter
                j+=1
            i+=1
            prev_d = cur_d
        B_inv = np.linalg.pinv(B)
        B_inv[np.abs(B_inv) < self.tol] = 0.0
        return B, B_inv

    def A(self, subspace):
        to_remove = set()
        for i in range(0,self.N,self.D):
            block = subspace[: , i:i+self.D]
            if(np.sum(np.abs(block)) < 0.01): # This 0.01 can be changed! 
                to_remove.add(int(i/self.D))
        mat = np.zeros(self.N).reshape((self.N,1))
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if(j in to_remove):
                    continue
                else:
                    for k in range(0, self.n*self.d[i],self.d[i]):
                        vec = np.concatenate((np.zeros(self.n*sum(self.d[:i]) + k), 
                                              self.data[i][j], 
                                              np.zeros(self.N-(self.n*sum(self.d[:i]) + k)-self.d[i])))
                        mat = np.insert(mat, -1, vec, axis = 1)
        mat = np.delete(mat, -1, axis=1)
        if(mat.shape==(self.N,0)):
            return mat
        csp = orth(mat)
        csp[np.abs(csp) < self.tol] = 0.0
        return csp


    def compute_wong_limit(self, B,B_inv):
        nullsp = null_space(B)    
        nullsp = np.array(nullsp)
        W = self.A(nullsp)
        max_iter = 10
        i = 0
        if(W.shape == (self.N,0)): # to check if the null space is trivial
            return W
        stop = False
        while(stop == False and i<= max_iter):
            Wpr = self.A(np.dot(B_inv, W))
            i+=1
            if(W.shape==Wpr.shape and np.allclose(W, Wpr)):
                return Wpr
            W = Wpr
        if(i> max_iter):
            print("max_iter exceeded. Returning null")
            return None
        return None 

    def W_limit_in_im_B(self, W_limit, B, B_inv):
        if(W_limit.shape == (self.N,0)):
            return False
        
        for i in range(len(W_limit[0])):
            b = W_limit[:,i]
            if(not self.check(B, B_inv, b)):
                return False
        return True 

    def check(self, B, B_inv, b):
        ncols = len(B_inv[0])
        if(ncols!= len(b)):
            print("The solution length does not match the length of cols. Aborting")
            return None
        x = np.dot(B_inv, b)
        check = np.dot(B,x)
        return np.allclose(check, b, atol=2.0)

    def mutate(self, M):
        for i in range(len(M)):
            s = random.choice(self.S)
            M[i] = s*M[i]
        M_inv = np.linalg.pinv(M)
        return M, M_inv
 

    def algP(self):
        B, B_inv = self.build_B_inv_matrix()
        hard_stop = 5
        counter = 0
        while counter <= hard_stop:
            W_limit = self.compute_wong_limit(B,B_inv)
            if self.W_limit_in_im_B(W_limit, B,B_inv):
                return  np.dot(B_inv, W_limit)
            else:
                B, B_inv = self.mutate(B) 
            counter+=1
        if(counter == hard_stop):
            print("Too many iterations. Start over please")
            return None
    
    def generate_labels(self, data, shrunk_subsp):
        Y = np.dot(shrunk_subsp, shrunk_subsp.T)
        idxs = []
        for j in range(self.m):
            idx = []
            for i in range(1,self.n+1):
                Ii = range((i-1)*self.D, i*self.D)
                for r in Ii:
                    if(not np.isclose(Y[r][r], 0)):
                        idx.append(i-1)
                        break 
            idxs.append(idx)
        labels = np.array([[1 if i in idxs[j] else 0 for i in range(len(data[j]))] for j in range(len(idxs))])
        return labels




