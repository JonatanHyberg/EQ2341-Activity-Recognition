import numpy as np

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True
            self.end_state = self.A.shape[0]


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        if self.is_finite:
            state = np.random.choice(self.nStates+1, 1, p=self.q)
            state_seq = [state]
            for _ in range(tmax-1):
                state = np.random.choice((self.nStates+1), 1, p = np.ravel(self.A[state,:]))
                if(state == self.nStates+1):
                    print("should break")
                if state == self.end_state:
                    break
                state_seq.append(state)
        else:
            state = np.random.choice(self.nStates, 1, p=self.q)
            state_seq = [state]
            for _ in range(tmax-1):
                state = np.random.choice(self.nStates,1, p=np.ravel(self.A[state,:]))
                state_seq.append(state)

        return np.array(state_seq)
    


    """ q_log = np.log(self.q)
        alpha_temp = q_log + pX[:,0]
        c1 = np.sum(alpha_temp)
        alpha_hat = alpha_temp / c1
        print(alpha_hat.shape)"""
    def forward(self, pX):

        len_seq = pX.shape[1]
        N = self.nStates

        alpha_hat = np.zeros((N, len_seq))
        if self.is_finite:
            c = np.zeros(len_seq+1)
        else:
            c = np.zeros(len_seq)

        #Init
        alpha_temp = self.q[:N] * pX[:,0]
        c[0] = np.sum(alpha_temp)
        alpha_hat[:,0] = alpha_temp / c[0]

        #calc 1 to T
        for t in range(1, len_seq):
            a = pX[:,t] * (alpha_hat[:,t-1].T @ self.A[:,:N])
            c[t] = np.sum(a)
            alpha_hat[:,t] = a/c[t]
            
        if self.is_finite:
            c[len_seq] = alpha_hat[:,-1] @ self.A[:,self.end_state]
            
        return alpha_hat, c

    
    def backward(self, c, pX):
        N = self.nStates
        T = pX.shape[1]
        beta_hat = np.zeros((N,T))

        if self.is_finite:
            beta_hat[:,-1] = self.A[:,N] / (c[-2]*c[-1])
            for t in reversed(range(T-1)):
                beta = self.A[:,:N] @ (pX[:,t+1] * beta_hat[:,t+1])
                beta_hat[:, t] = beta / c[t]

        else:
            beta_hat[:,-1] = 1/c[-1]
            for t in reversed(range(T-1)):
                beta = self.A[:,:N] @ (pX[:,t+1] * beta_hat[:,t+1])
                beta_hat[:, t] = beta / c[t]

        return beta_hat

