"""
Evaluation metrics
"""

import numpy as np
from scipy import linalg
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_mse(
  ts_true,
  ts_pred
):
  """Mean squarred error"""

  return mean_squared_error(real_data.flatten(), generated_data.flatten())

def compute_mae(
  ts_true,
  ts_pred
):
  """Mean absolute error"""

  return mean_absolute_error(real_data.flatten(), generated_data.flatten())

def compute_dtw_distance(
  ts_true, 
  ts_pred
):
  """Dynamic Time Warping distance"""

  from scipy.spatial.distance import euclidean
    
    n, m = len(ts1), len(ts2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(ts1[i-1], ts2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                         dtw_matrix[i, j-1],
                                         dtw_matrix[i-1, j-1])
    
    return dtw_matrix[n, m]

def compute_fid(
  ts_true, 
  ts_pred
):
  """Frechet Inception Distance"""

  mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
  mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
  
  ssdiff = np.sum((mu1 - mu2)**2.0)
  covmean = linalg.sqrtm(sigma1.dot(sigma2))
  
  if np.iscomplexobj(covmean):
      covmean = covmean.real
  
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

  return fid

def compute_mmd(
  ts_true, 
  ts_pred, 
  kernel='rbf', 
  gamma=1.0
):
    """Maximum Mean Discrepancy"""
  
    def rbf_kernel(X, Y, gamma):
        XX = np.sum(X**2, axis=1)[:, None]
        YY = np.sum(Y**2, axis=1)[None, :]
        XY = X @ Y.T
        distances = XX + YY - 2*XY
        return np.exp(-gamma * distances)
    
    real_flat = ts_true.reshape(len(ts_true), -1)
    gen_flat = ts_pred.reshape(len(ts_pred), -1)
    
    K_XX = rbf_kernel(real_flat, real_flat, gamma)
    K_YY = rbf_kernel(gen_flat, gen_flat, gamma)
    K_XY = rbf_kernel(real_flat, gen_flat, gamma)
    
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
  
    return mmd
