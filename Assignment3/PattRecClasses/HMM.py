import numpy as np
from .GaussD import GaussD
from .MarkovChain import MarkovChain


class HMM:
    """
    HMM - class for Hidden Markov Models, representing
    statistical properties of random sequences.
    Each sample in the sequence is a scalar or vector, with fixed DataSize.
    
    Several HMM objects may be collected in a single multidimensional array.
    
    A HMM represents a random sequence(X1,X2,....Xt,...),
    where each element Xt can be a scalar or column vector.
    The statistical dependence along the (time) sequence is described
    entirely by a discrete Markov chain.
    
    A HMM consists of two sub-objects:
    1: a State Sequence Generator of type MarkovChain
    2: an array of output probability distributions, one for each state
    
    All states must have the same class of output distribution,
    such as GaussD, GaussMixD, or DiscreteD, etc.,
    and the set of distributions is represented by an object array of that class,
    although this is NOT required by general HMM theory.
    
    All output distributions must have identical DataSize property values.
    
    Any HMM output sequence X(t) is determined by a hidden state sequence S(t)
    generated by an internal Markov chain.
    
    The array of output probability distributions, with one element for each state,
    determines the conditional probability (density) P[X(t) | S(t)].
    Given S(t), each X(t) is independent of all other X(:).
    
    
    References:
    Leijon, A. (20xx) Pattern Recognition. KTH, Stockholm.
    Rabiner, L. R. (1989) A tutorial on hidden Markov models
    	and selected applications in speech recognition.
    	Proc IEEE 77, 257-286.
    
    """
    def __init__(self, mc, distributions, outputWeights):

        self.stateGen = mc
        self.outputDistr = distributions

        self.outoutWeights = outputWeights

        self.nStates = mc.nStates
        self.dataSize = distributions[0][0].dataSize
    
    def rand(self, nSamples):
        """
        [X,S]=rand(self,nSamples); generates a random sequence of data
        from a given Hidden Markov Model.
        
        Input:
        nSamples=  maximum no of output samples (scalars or column vectors)
        
        Result:
        X= matrix or row vector with output data samples
        S= row vector with corresponding integer state values
          obtained from the self.StateGen component.
          nS= length(S) == size(X,2)= number of output samples.
          If the StateGen can generate infinite-duration sequences,
              nS == nSamples
          If the StateGen is a finite-duration MarkovChain,
              nS <= nSamples
        """
        
        #*** Insert your own code here and remove the following error message 
        states = self.stateGen.rand(nSamples)
        outputs = []
       
        for state in np.ravel(states):
            state_distro = np.random.choice(a=self.outputDistr[state], p=self.outoutWeights[state])
            outputs.append(state_distro.rand(1))

        return np.array(outputs), states
        
    def viterbi(self, X):

        pX = np.zeros((3,X.shape[0]))
        for i, distrobution in enumerate(self.outputDistr):
            pX[i,:] = distrobution.prob(X[:,:])
        pX = pX / np.max(pX,axis=0)



        xi = np.zeros((self.nStates, pX.shape[1]))
        back_pointers = np.zeros((self.nStates,pX.shape[1]))

        #FORWAD
        for i in range(self.nStates):
            xi[i,0] = self.stateGen.q[i] * pX[0,i]
        for t in range(1,pX.shape[1]):
            for j in range(self.nStates):
                xi[j,t] = pX[j,t] * np.max(xi[:,t-1]*self.stateGen.A[:,j])
                back_pointers[j,t] = (np.argmax(xi[:,t-1]*self.stateGen.A[:,j]))
        
        #BACKPASS
        state_seq = []
        state_seq.append(np.argmax(xi[:,-1]))

        for t in reversed(range(pX.shape[1]-1)):

            it = int(back_pointers[state_seq[-1],t+1])
            state_seq.append(it)
        state_seq.reverse()

        return np.array(state_seq)
    
    def log_viterbi(self, X):

        pX = np.zeros((3,X.shape[0]))
        for i, distrobution in enumerate(self.outputDistr):
            for j in range(len(distrobution)):
                pX[i,:] = self.outoutWeights[i][j] * distrobution[j].prob(X[:,:]) + np.finfo(np.float64).eps

        pX = np.log(pX) #/ np.max(pX,axis=0)


        xi = np.zeros((self.nStates, pX.shape[1]))
        back_pointers = np.zeros((self.nStates,pX.shape[1]))

        #FORWAD
        for i in range(self.nStates):
            xi[i,0] = np.log(self.stateGen.q[i]) + pX[0,i]

        for t in range(1,pX.shape[1]):
            for j in range(self.nStates):
                xi[j,t] = pX[j,t] + np.max(xi[:,t-1]+ np.log(self.stateGen.A[:,j]))
                back_pointers[j,t] = (np.argmax(xi[:,t-1]+np.log(self.stateGen.A[:,j])))
        
        #BACKPASS
        state_seq = []
        state_seq.append(np.argmax(xi[:,-1]))

        for t in reversed(range(pX.shape[1]-1)):

            it = int(back_pointers[state_seq[-1],t+1])
            state_seq.append(it)
        state_seq.reverse()

        return np.array(state_seq)



    def train(self, X): 
        #These are list of length R
        alpha_hat_list = []
        c_list = []
        beta_hat_list = []
        gamma_list = []
        gamma_m_list = []
        xi_list = []




        #RUNS foward and backward algorithm over entire Dataset
        for r in range(len(X)):
            #calc output probability
            pX = np.zeros((3,X[r].shape[0]))
            for i, distrobution in enumerate(self.outputDistr):
                for m in range(len(distrobution)):
                    pX[i,:] += self.outoutWeights[i][m] * distrobution[m].prob(X[r][:,:])
            pX = pX + np.finfo(np.float64).tiny
            unscaled_px = np.copy(pX) 
            
            pX = pX / np.max(pX,axis=0)

            #Foward algorithm
            alpha_hat, c = self.stateGen.forward(pX)
            alpha_hat_list.append(alpha_hat)
            c_list.append(c)

            #backward Algorithm
            beta_hat = self.stateGen.backward(c, pX)
            beta_hat_list.append(beta_hat)


            #Calc Gamma:
            gamma = alpha_hat * beta_hat * c
            gamma_list.append(gamma)

            #Calc Gamma m:
            gamma_m = np.zeros((self.nStates,len(self.outoutWeights[2]),X[r].shape[0]))

            for i in range(self.nStates):
                for m in range(len(self.outoutWeights[i])):
                    for t in range(X[r].shape[0]):
                        gamma_m[i,m,t] = gamma_list[r][i,t] * self.outoutWeights[i][m]*self.outputDistr[i][m].prob(X[r][t,:]) / unscaled_px[i,t]
            gamma_m_list.append(gamma_m)


            #Calc Xi
            xi = np.zeros((self.nStates,self.nStates,X[r].shape[0]-1))
            for t in range(X[r].shape[0]-1):
                for i in range(self.nStates):
                    for j in range(self.nStates):
                        xi[i,j,t] = alpha_hat[i,t] * self.stateGen.A[i,j] * pX[j,t+1] * beta_hat[j,t+1]
            #Sum over time
            xi = np.sum(xi,axis=2)
            xi_list.append(xi)
        

        #Baum Welch Algorithm

        #calc new q
        gamma_sum = np.zeros((self.nStates,1))
        for gamma in gamma_list:
            gamma_sum = gamma_sum + gamma[:,0][:,np.newaxis]
        q_new = gamma_sum / np.sum(gamma_sum,axis=0)
        
        #Calc new A

        #Sum over R
        xi_line = np.zeros((self.nStates,self.nStates))
        for xi in xi_list:
            xi_line = xi_line + xi

        A_new = xi_line / np.sum(xi_line,axis=1)[:,np.newaxis]

        
        #Calc new output Weights

        numerator = np.zeros((self.nStates,len(self.outoutWeights[2]),1))
        denominator = np.zeros((self.nStates,1,1))

        for r in range(len(gamma_m_list)):
            for i in range(self.nStates):
                for m in range(len(self.outoutWeights[i])):
                    for t in range(X[r].shape[0]):
                        numerator[i,m,:] = numerator[i,m,:] + gamma_m_list[r][i,m,t] 
                        denominator[i,0,0] = denominator[i,0,0] + gamma_m_list[r][i,m,t]

        weight_new = np.zeros((self.nStates,len(self.outoutWeights[2]),1))
        for i in range(self.nStates):
            for m in range(len(self.outoutWeights[i])):
                weight_new[i,m,:] = numerator[i,m,:] / denominator[i,0,0]


        #calc Mu

        numerator = np.zeros((self.nStates,len(self.outoutWeights[2]),self.nStates))
        denominator = np.zeros((self.nStates,len(self.outoutWeights[2]),1))

        for r in range(len(gamma_m_list)):
            for i in range(self.nStates):
                for m in range(len(self.outoutWeights[i])):
                    for t in range(X[r].shape[0]):
                        numerator[i,m,:] = numerator[i,m,:] + gamma_m_list[r][i,m,t] * X[r][t,:] 
                        denominator[i,m,0] = denominator[i,m,0] + gamma_m_list[r][i,m,t]

        mu_new = np.zeros((self.nStates,len(self.outoutWeights[2]),self.nStates))

        for i in range(self.nStates):
            for m in range(len(self.outoutWeights[i])):             
                mu_new[i,m,:] = numerator[i,m,:] / denominator[i,m]


        #Calc Covariance matrix:

        cov_new = np.zeros((self.nStates,len(self.outoutWeights[2]), self.nStates, self.nStates))
        numerator = np.zeros((self.nStates,len(self.outoutWeights[2]),self.nStates,self.nStates))
        for r in range(len(gamma_m_list)):
            for i in range(self.nStates):
                for m in range(len(self.outoutWeights[i])):
                    for t in range(X[r].shape[0]):
                        numerator[i,m,:,:] = numerator[i,m,:,:] + gamma_m_list[r][i,m,t] * ((X[r][t,:]-mu_new[i,m,:])[:,np.newaxis] @ (X[r][t,:]-mu_new[i,m,:])[np.newaxis,:])


        for i in range(self.nStates):
            for m in range(len(self.outoutWeights[i])):
                cov_new[i,m,:,:] = numerator[i,m,:,:] / denominator[i,m]
    
        cov_new = cov_new / cov_new.max()

        
        #Update Variables

        self.stateGen.q = q_new[:,0]
        self.stateGen.A = A_new
        for i in range(self.nStates):
            for m in range(len(self.outoutWeights[i])):
                self.outoutWeights[i][m] = weight_new[i,m,0]
        for i in range(self.nStates):
            for m in range(len(self.outoutWeights[i])):
                self.outputDistr[i][m] = GaussD(mu_new[i,m,:],cov_new[i,m,:,:])
        


        return

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass

    def logprob(self, values):
        probability_matrix = np.zeros((self.nStates, values.shape[0]))
        for i in range(self.nStates):
            prob = self.outputDistr[i].prob(values[:,:])
            probability_matrix[i] = prob

        _, c = self.stateGen.forward(probability_matrix)
        
        return np.sum(np.log(c))

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass