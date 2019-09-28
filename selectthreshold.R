selectThreshold <- function(yval, pval) {
  bestEpsilon <- 0
  bestF1 <- 0
  F1 <- 0
  
  stepsize <- (max(pval) - min(pval)) / 1000
  for (epsilon in seq(min(pval),max(pval),stepsize)) {   
    predictions <- (pval < epsilon)
    tp <- sum((yval == 1) & (predictions == 1))
    fp <- sum((yval == 0) & (predictions == 1))
    fn <- sum((yval == 1) & (predictions == 0))
    
    prec <- tp / (tp + fp)
    rec  <- tp / (tp + fn)
    
    F1 <- 2 * prec * rec / (prec + rec)  
      
    if (!is.na(F1) && !is.na(bestF1) && F1 > bestF1) {
      bestF1 <- F1
      bestEpsilon <- epsilon
    }
  }
  list(bestEpsilon = bestEpsilon, bestF1 = bestF1)
}