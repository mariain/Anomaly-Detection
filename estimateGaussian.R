estimateGaussian <- function(X) {
  #ESTIMATEGAUSSIAN This function estimates the parameters of a
  #Gaussian distribution using the data in X
  #   mu_sigma2 <- estimateGaussian(X),
  #   The input X is the dataset with each n-dimensional data point in one row
  #   The output is an n-dimensional vector mu, the mean of the data set
  #   and the variances sigma^2, an n x 1 vector
  #
  
  m <- dim(X)[1]
  n <- dim(X)[2]
  
  mu <- rep(0,n)
  sigma2 <- rep(0,n)
  
  mu <- 1 / m * apply(X,2,sum)
  sigma2 <- apply(X,2,var) * (m - 1) / m
  
  list(mu = mu, sigma2 = sigma2)
}
