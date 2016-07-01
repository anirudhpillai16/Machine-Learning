data=read.csv('C:/ML/Assignment 3/a3barebones/susysubset.csv')
mydata<- data[sample(1:nrow(data),2700,replace = FALSE),]
# Split Data set
sub <- sample(nrow(mydata),floor(nrow(mydata) * 0.75))
# Spliting Training and Test Data in ratio 3:1
training <- mydata[sub, ]
testing <- mydata[-sub, ]
# X train and Y train
n=length(training)
xtrain<- training[,-n]
ytrain<- training[,n]
# X test and Y test
xtest<- testing[,-n]
predictions<- testing[,n]
# Adding Colummn of 1's to X test and X train
xtrain$newcol<-rep(1,nrow(xtrain))
xtest$newcol<-rep(1,nrow(xtest))
# Initialize weights as Null
weight=c()
regwt=0.01
# Replace 0 by -1
yt=ytrain
yt[yt=='0']=-1
c<- xtrain[sample(1:nrow(xtrain),40,replace=FALSE),]
k=c()
ct=t(c)
# Quadratic Kernel (Training)
sigma=0.5
tt1=1/(2*(sigma^2))
nt=nrow(xtrain)
for( i in 1:nt) {
  for(j in 1:nrow(c)){
    tt2=xtrain[i,]-c[j,]
    tt3=tt2^2
  }
}
k= exp(-tt1*tt3)
numsamples=ncol(k)
kt=t(k)
term1=kt%*%k/numsamples
term2=regwt*diag(numsamples)
term3= term1+term2
term4=solve(term3)
term5=kt%*%yt
weight=term4%*%term5
# Quadratic Kernel (1+X*cT)^2  (Test)
nte=nrow(xtest)
for( i in 1:nte) {
  for(j in 1:nrow(ct)){
    t2=xtest[i,]-ct[j,]
    t3=tt2^2
  }
}
Ktest= exp(-tt1*t3)


ytest = as.matrix(Ktest)%*%as.matrix(weight)
ytest[ytest>0]=1
ytest[ytest<0]=0
correct=0
for(i in 1:nrow(ytest)){
  if(predictions[i]==ytest[i]){
    correct = correct + 1
  }
}
acc = correct/length(ytest)*100
print("Accuracy ")
acc
