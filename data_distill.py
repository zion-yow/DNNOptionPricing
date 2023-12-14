import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
import multiprocessing as mp

class Data_distill:
    def __init__(self) -> None:
        self.S = np.arange(1,101)
        self.K = np.arange(1,101)
        self.sigma = np.arange(1,51,10)/100
        self.T = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 1, 5])
        self.r = np.arange(1,11)/100
        

    def bs_generate(self, T, S, sigma, r, K):
        '''
        S: spot price
        K: strike price
        T: time to maturity
        r: risk-free interest rate
        sigma: standard deviation of price of underlying asset
        '''
        d1 = ( np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        #  call option prcie
        C = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r*T) * norm.cdf(d2, 0.0, 1.0))

        return C


    def bnt_generate(self, N,T,S0,sigma,r,K,show_array=False):
        '''
        N: number of time interval (number of tree seperation)
        '''
        dt=T/N
        u=np.exp(sigma*np.sqrt(dt))
        d=1/u
        p=(np.exp(r*dt)-d)/(u-d)# 贴现

        price_tree=np.zeros([N+1,N+1])
        for i in range(N+1):
            for j in range(i+1):
                price_tree[j,i]=S0*(d**j)*(u**(i-j))
        # pick ITM
        option_tree=np.zeros([N+1,N+1])
        option_tree[:,N]=np.maximum(np.zeros(N+1),price_tree[:,N]-K)

        for i in np.arange(N-1,-1,-1): # [2,1,0]
            for j in np.arange(0,i+1):# [0,1,2]
                option_tree[j,i]=np.exp(-r*dt)*(p*option_tree[j,i+1]+(1-p)*option_tree[j+1,i+1])

        if show_array:
            return [option_tree[0,0],np.round(price_tree),np.round(option_tree)]
        else:
            return option_tree[0,0]


    def mc_generate(self,T,S0,sigma,r,K):
        np.random.seed(522)
        random_Z =  np.random.randn(1000)
        
        def discount_ST(randZ):
            St = S0*np.exp((r-0.5*sigma**2)*T + randZ*sigma*np.sqrt(T))
            St_discount = np.exp(-r*T)*np.max(St-K,0)
            return St_discount

        STs = np.apply_along_axis(discount_ST,0,random_Z)
        C = np.nanmean(STs)
        return C

    def main(self):
        generating_arr = {}
        for s in tqdm(self.S):
            for k in self.K:
                for sig in self.sigma:
                    for t in self.T:
                        for r in self.r:
                            generating_arr[f'S{s}_K{k}_sigma{sig}_T{t}_r{r}']= np.array([t, s, sig, r, k, 
                            self.bs_generate( t, s, sig, r, k),
                            self.bnt_generate(10,t,s,sig,r,k),
                            self.mc_generate(t,s,sig,r,k)])
        df = pd.DataFrame(generating_arr,index = ['T','S0','sigma','r','K','BS','BNT','MC']).T
        return df


if __name__ == '__main__':
    oj = Data_distill()
    df = oj.main()
    df.to_pickle('distilled_data.pkl')