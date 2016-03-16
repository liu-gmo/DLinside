sigmoid <- function(z) {
  return(1/(1 + exp(-z)))
}

## Loss Function (Cross Entropy)
forward <- function(theta, X, y, lambda) {
  m <- length(y)
  return(1/m * sum(-y * log(sigmoid(X %*% theta)) - (1 - y) *
                     log(1 - sigmoid(X %*% theta))) + lambda/2/m * sum(theta[-1]^2))
}


## Gradient
gradient <- function(theta, X, y, lambda) {
  m <- length(y)
  return(1/m * t(X) %*% (sigmoid(X %*% theta) - y) + lambda/m *
           c(0, theta[-1]))
}

# expand 2-dimension to N dimensions: N = (degree + 1) * (degree + 2) / 2
mapFeature <- function(X1, X2, degree) {
  out <- rep(1, length(X1)) #for bias
  for (i in 1:degree) {
    for (j in 0:i) {
      out <- cbind(out, (X1^(i - j)) * (X2^j))
    }
  }
  return(out)
}


lambda <- 1
learningRate <- 1e-1
maxIteration <- 10000
tolerance <- 1e-8

mu <- 0.9
v <- NA
m <- NA
cache <- NA
decayRate <- 0.1
beta1 <- 0.9
beta2 <- 0.995

train <- function(dt, y, method){
  nPara <- ncol(dt)
  initialTheta <- rep(0, nPara)
  v <<- rep(0, nPara)
  m <<- rep(0, nPara)
  cache <<- rep(0, nPara)
  
  #sampling data from all train data: mini-batch
  #here, give all data (full batch)
  res <- paraOptim(dt, y, initialTheta, method)
}

TRACE_MODE <- TRUE
trace.log <- NULL
trace.level <- 10

paraOptim <- function(dt, y, theta, method){
  trace.df <- NULL
  
  pre_loss <- NaN
  for(i in 0:maxIteration){
    loss <- forward(theta, dt, y, lambda)
    dx <- gradient(theta, dt, y, lambda)
    
    if(TRACE_MODE && (i %% trace.level == 0) ){
      trace.row <- c(epoch=i, loss=loss, theta=theta)
      trace.df <- rbind(trace.df, trace.row)
    }
      
    delta <- paraUpdate(method, dx, i+1)
    theta <- theta + delta
    
    #if(sqrt(sum(delta ^ 2)) < tolerance) break
    if((!is.nan(pre_loss)) && abs(pre_loss - loss) < tolerance) break
    pre_loss <- loss
  }
  
  if(i %% maxIteration != 0){
    trace.row <- c(epoch=i, loss=loss, theta=theta)
    trace.df <- rbind(trace.df, trace.row)
  }
  
  converged <- if(i >= maxIteration) FALSE else TRUE
  trace.log <<- rbind(trace.log, list(method=method, final.loss=loss, final.iteration=i, converged=converged, trace=trace.df))
  
  print(theta)
  print(i)
  print(loss)
  
  return(list(theta=theta,loss=loss,iteration=i))
}

paraUpdate <- function(method, dx, t){
  if(method == "SGD"){ #Stochastic Gradient Descent
    delta <- -learningRate * dx
  }
  else if(method == "momentum"){
    v <<- mu * v - learningRate * dx
    delta <- v
  }
  else if(method == "NAG"){ #Nesterov Accelerated Gradient
    v_prev <- v
    v <<- mu * v - learningRate * dx
    delta <- -mu * v_prev + (1 + mu) * v
  }
  else if(method == "AdaGrad"){ #Duchi et al., 2011
    cache <<- cache + dx ^ 2
    delta <- -learningRate * dx / (sqrt(cache) + 1e-7)
  }
  else if(method == "RmsProp"){ #Geoff Hinton’s Coursera, 2012
    cache <<- cache * decayRate + dx ^ 2 * (1 - decayRate)
    delta <- -learningRate * dx / (sqrt(cache) + 1e-7)
  }
  else if(method == "Adam"){ #Kingma and Ba, 2014
    m <<- beta1 * m + (1-beta1) * dx
    v <<- beta2 * v + (1-beta2) * (dx ^ 2)
    if(t < 10){
      mb <- m / (1-beta1^t) 
      vb <- v / (1-beta2^t) 
    }
    else{
      mb <- m
      vb <- v
    }
    delta <- -learningRate * mb / (sqrt(vb) + 1e-7)
  }
  else{
    print("Error: No such optimization method!")
  }
  
  return(delta)
}

# ***************************** TEST Stage ***************************

data <- read.csv("ex2data2.txt", header = F) #from Andrew Ng Course

sampleSize <- 1000

randomSample <- function(sampleSize){
  data <- cbind(runif(sampleSize, -1, 1), runif(sampleSize, -1, 1), runif(sampleSize, -1, 1))
  data[,3] <- ifelse(data[,3] > 0, 1, 0)
  return(data)
}

#data <- randomSample(sampleSize) #generate random data

X <- as.matrix(data[,c(1,2)])
y <- data[,3]
X <- mapFeature(X[,1],X[,2], 3)
m <- nrow(X)
n <- ncol(X)

initialTheta <- rep(0, n)
#initialTheta <- rnorm(n) / sqrt(n) #Xavier initialization (Glorot et al., 2010)

trace.log <- NULL

train(X, y, "SGD")
train(X, y, "momentum")
train(X, y, "NAG")
train(X, y, "AdaGrad")
train(X, y, "RmsProp")
train(X, y, "Adam")

res <- optim(initialTheta, forward, gradient, X, y, lambda, method = "BFGS", control = list(maxit = maxIteration, trace=3))
res$par
res$value
res$counts
res <- optim(initialTheta, forward, gradient, X, y, lambda, method = "L-BFGS-B", control = list(maxit = maxIteration, trace=6))
res$par
res$value
res$counts

# **************************** visulization ****************************

plotRes <- function(type){
  #color_arr <- NULL;
  xrange <- NA
  yrange <- NA
  labels <- NULL;
  for(i in 1:nrow(trace.log)){
    col <- rgb(runif(5),runif(5),runif(5))
    #color_arr <- cbind(color_arr, col)
    
    labels <- c(labels, trace.log[[i,1]])
    
    xx <- trace.log[i,5]
    td <- as.data.frame(xx$trace)
    if(is.na(xrange)) {
      xrange <- c(min(td$theta2),max(td$theta2))
    } 
    else{
      xrange = c(min(xrange[1], min(td$theta2)),max(xrange[2], max(td$theta2)))
    }
    
    if(is.na(yrange)){
      yrange <- c(min(td$theta3),max(td$theta3))
    }
    else{
      yrange = c(min(yrange[1], min(td$theta3)),max(yrange[2], max(td$theta3)))  
    }
  }
  
  xx <- trace.log[1,5]
  td <- as.data.frame(xx$trace)
  if(type == "epoch"){
    px <- td$epoch
    py <- td$loss  
    
    plot(x=px, y=py, col=color_arr[1], type="l", main="Convergence Performance", pch=0, xlab="epoch", ylab="loss", lty=1, lwd=2, xlim=c(0,300))
  }
  else{
    px <- td$theta2
    py <- td$theta3  
    
    xrange[2] <- xrange[2] + (xrange[2]-xrange[1])*0.2 #preserve space for legend
    plot(x=px, y=py, col=color_arr[1], type="l", main="Convergence Performance", pch=0, xlab="para 1", ylab="para2", lty=1, lwd=2,xlim=xrange,ylim=yrange)
    
    points(x=px[1], y=py[1], col="black", type='p', lwd=2, lty=1, pch=16, cex=2)  
  }
  
  for(i in 2:nrow(trace.log)){
    xx <- trace.log[i,5]
    td <- as.data.frame(xx$trace)
    
    if(type == "epoch"){
      px <- td$epoch
      py <- td$loss
    }
    else{
      px <- td$theta2
      py <- td$theta3
    }
    points(x=px, y=py, col=color_arr[i], type='l', lwd=2, lty=1, pch=0)  
    
    fp <- nrow(td)
    points(x=px[fp], y=py[fp], col=color_arr[i], type='p', lwd=2, lty=1, pch=13,cex=1.5)  #final point
  }
  
  legend("topright", legend = labels, col = color_arr, 
         lty = 1, bg = "transparent", lwd=2)
  
}

plotRes("epoch")
plotRes("position")
