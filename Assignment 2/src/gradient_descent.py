x=[-1.2,1]
f = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2;
alpha = 0.0005
grad=[1.0,1.0]
count=0;

while (grad[0]<-0.00000015 or grad[0]>0.00000015) or (grad[1]<-0.00000015 or grad[1]>0.00000015) :
	grad[0]=-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0])
	grad[1]=200*(x[1]-x[0]**2)
	x[0]=x[0]-alpha*grad[0]
	x[1]=x[1]-alpha*grad[1]
	f = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2;
	print x[0], grad[0], x[1], grad[1], f;
	count=count+1

print "{x0,x1,grad0,grad1}"
print "Number of Iterations = "
print count


