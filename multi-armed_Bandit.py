import torch as th
from torch import nn
from torch.distributions.categorical import Categorical



class bandit_arg:
    def __init__(self,K):
        self.K = K
        self.L = th.zeros(self.K)
        self.eta = 0.05
    def pick(self,t):
        self.pt =  nn.Softmax(dim = 0)( -self.eta * self.L )
        m = Categorical( self.pt )
        self.at = m.sample()
        return self.at
                 
    def update(self, lt):
        self.L[ self.at ] += lt/self.pt[ self.at ]


class adversary:
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.L = th.zeros( self.T, self.K )
        self.m = Categorical( th.tensor([ i+1.0 for i in range(self.K)] ))
    
    def process(self, t):
        self.chosen = self.m.sample()
        self.L[ t ][ self.chosen ] = 1

    def loss( self, at, t):
        #return 1 if at < 3 else 0
        return self.L[ t ][ at ]



T = 100000
K = 5

A = bandit_arg(K)
model = adversary(K, T)

slt = 0

pwr = [ (i**2)-1 for i in range(3,1001) ]

for t in range(T):
    at = A.pick( t )
    model.process( t )
    lt = model.loss( at, t)
    slt += lt
    A.update( lt )

print("regret:", float ( slt - th.sum(model.L,0)[0] ) )
    



