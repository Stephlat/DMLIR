import numpy as np

_LOG_2PI = np.log(2 * np.pi)

# log of pdf for gaussian distributuion with diagonal covariance matrix
def loggausspdf(X, mu, cov):
    if len(X.shape)==1:
        D=1
    else:
        D = X.shape[1]
    
    logDetCov = D*np.log(cov)
    dxM = X - mu
    L = np.sqrt(cov)
    xRinv = 1/L * dxM
    mahalaDx = np.sum(xRinv**2, axis=1)
    y = - 0.5 * (logDetCov + D*_LOG_2PI + mahalaDx)
    return y

def gausspdf(X, mu, cov):
    return np.exp(loggausspdf(X, mu, cov))

# log of pdf for gaussian distributuion with full covariance matrix (cholesky factorization for stability)
def chol_loggausspdf(X, mu, cov):

    D = X.shape[0]
    
    X = X - mu #DxN
    U = np.linalg.cholesky(cov).T #DxD
    Q = np.linalg.solve(U.T,X)
    q = np.sum(Q**2, axis=0)
    c = D*_LOG_2PI + 2*np.sum(np.log(np.diag(U)))
    y = -0.5 * (c + q)

    return y 
