import numpy as np
from lab2_tools import logsumexp


# already implemented
def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    num_states_hmm1 = len(hmm1['startprob'])-1
    num_states_hmm2 = len(hmm2['startprob'])-1
    num_states_concat = num_states_hmm1+num_states_hmm2+1
    
    startprob = np.concatenate((hmm1['startprob'],hmm2['startprob'][1:]))

    transmat = np.zeros((num_states_concat,num_states_concat))
    transmat[:num_states_hmm1+1,:num_states_hmm1+1] = hmm1['transmat']
    transmat[num_states_hmm1:,num_states_hmm1:] = hmm2['transmat']

    means = np.concatenate((hmm1['means'], hmm2['means']), axis=0)

    covars = np.concatenate((hmm1['covars'], hmm2['covars']), axis=0)

    concatenated_hmm = {'startprob': startprob,
                       'transmat': transmat,
                       'means': means,
                       'covars': covars}
    
    return concatenated_hmm


# already implemented, uses concatTwoHMMs()
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape  # N time frames, M states
    log_alpha = np.zeros((N, M))

    # Initialization step
    log_alpha[0] = log_startprob[:M] + log_emlik[0]

    # Recursion step
    for t in range(1, N):
        for j in range(M):
            log_alpha[t, j] = logsumexp(log_alpha[t - 1] + log_transmat[:M, j]) + log_emlik[t, j]

    return log_alpha


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    log_beta = np.zeros((N, M))

    # Initialization at last time step: log(1) = 0
    log_beta[-1, :] = 0

    # Recursion: move backward from T-2 to 0
    for t in range(N - 2, -1, -1):
        for i in range(M):
            log_beta[t, i] = logsumexp(
                log_transmat[i, :M] + log_emlik[t + 1, :] + log_beta[t + 1, :]
            )

    return log_beta

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = log_emlik.shape
    logdelta = np.full((N, M), -np.inf)  # Viterbi log-likelihoods
    psi = np.zeros((N, M), dtype=int)    # backpointers

    # Initialization
    logdelta[0] = log_startprob[:M] + log_emlik[0]

    # Recursion
    for t in range(1, N):
        for j in range(M):
            scores = logdelta[t - 1] + log_transmat[:M, j]
            psi[t, j] = np.argmax(scores)
            logdelta[t, j] = np.max(scores) + log_emlik[t, j]

    # Termination
    if forceFinalState:
        last_state = M - 1
        viterbi_loglik = logdelta[-1, last_state]
    else:
        last_state = np.argmax(logdelta[-1])
        viterbi_loglik = logdelta[-1, last_state]

    # Backtracking
    viterbi_path = np.zeros(N, dtype=int)
    viterbi_path[-1] = last_state
    for t in range(N - 2, -1, -1):
        viterbi_path[t] = psi[t + 1, viterbi_path[t + 1]]

    return viterbi_loglik, viterbi_path

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    log_gamma = log_alpha + log_beta  # element-wise add

    # Normalize each row with logsumexp to get valid probabilities (in log domain)
    for t in range(log_gamma.shape[0]):
        log_gamma[t] -= logsumexp(log_gamma[t])  # subtract log-sum-exp to normalize

    return log_gamma

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    # Convert from log domain to linear domain
    gamma = np.exp(log_gamma)

    # Number of states (M) and feature dimension (D)
    N, D = X.shape
    M = gamma.shape[1]

    # Initialize outputs
    means = np.zeros((M, D))
    covars = np.zeros((M, D))

    # For each state i
    for i in range(M):
        # Gamma weights for state i
        gamma_i = gamma[:, i]  # Shape (N,)

        # Denominator for normalization
        gamma_sum = np.sum(gamma_i)

        # Weighted mean
        means[i] = np.sum(gamma_i[:, np.newaxis] * X, axis=0) / gamma_sum

        # Weighted variance
        diff = X - means[i]
        covars[i] = np.sum(gamma_i[:, np.newaxis] * diff**2, axis=0) / gamma_sum

    # Apply variance floor
    covars = np.maximum(covars, varianceFloor)

    return means, covars