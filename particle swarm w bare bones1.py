import numpy as np;
import random
import math
%matplotlib nbagg
import matplotlib.pyplot as plt

alg="bb"  #"canon"
popsize=20;
dimen=30;
iters=1000;
searchspace=10;
offset=20;
w1=0.7298;
w2=1.49618;
pbest=np.zeros(popsize);
nbr=np.zeros((popsize,2), dtype=np.int );
best=np.zeros(iters);

def Sphere(stuff):
    ans=0.0;
    for j in range(0,dimen):
        ans=ans+(stuff[j]-offset)**2;
    return ans;

def Griewank(chromosome):
    part1 = 0
    for i in range(len(chromosome)):
        part1 += (chromosome[i]-offset)**2
        part2 = 1
    for i in range(len(chromosome)):
        part2 *= math.cos(float((chromosome[i]-offset)) / math.sqrt(i+1))
    return 1 + (float(part1)/4000.0) - float(part2)

eval=Griewank;

#Make Lbest neighborhoods;
for i in range(0,popsize):
    nbr[i,0]=i-1;
    nbr[i,1]=i+1;
nbr[0,0]=popsize-1;
nbr[popsize-1,1]=0;
print("Neighborhoods");    
for i in range(0,popsize):
    print(i, nbr[i,0], nbr[i,1])

x = np.random.random( size=(popsize, dimen))*2.0*searchspace - searchspace;
v = np.random.random( size=(popsize, dimen))*searchspace - searchspace/2;
p = np.random.random( size=(popsize, dimen))*2.0*searchspace - searchspace;
#First time through;
for i in range(0,popsize):
    pbest[i]=(eval(x[i]));
gbest=0;  # Arbitrary assignment;

print("Initial population");
for i in range(0,popsize):
    print(i, x[i], ' Eval= ', pbest[i]);

#Having created p and pbest, now reinitialize exes and go;

for pso in range(0,iters):
    for i in range(0,popsize):        
        g=nbr[i,0];
        if pbest[nbr[i,1]] < pbest[g]:
            g=nbr[i,1];
        for j in range(0,dimen):
            if alg=="canon":
                v[i,j] = w1*v[i,j]  \
                + random.random()*w2*(p[i,j]-x[i,j]) \
                + random.random()*w2*(p[g,j]-x[i,j]);
                x[i,j]=x[i,j]+v[i,j];
            else:
                sd=abs(p[i,j]-p[g,j]);
                mid=(p[i,j]+p[g,j])/2;
                x[i,j]=random.gauss(mid, sd);

        xeval=eval(x[i]);
        if xeval < pbest[i]:
            pbest[i]=xeval
            p[i]=x[i];
            if xeval < pbest[gbest]:
                gbest=i; 
    best[pso]=math.log10(pbest[gbest]);           
    #best[pso]=pbest[gbest]; 
print;
print;
print("Final values");
for i in range(0,popsize):
    print(i, p[i], ' Eval= ', pbest[i]);
print; 
print;
print('Algorithm is', alg)
print;
print("Best function result is ", pbest[gbest]);
print;
plt.plot(best, 'r-', linewidth='3');
plt.title("Function result over time");
plt.ylabel("(Log) Error");
plt.xlabel("Iterations");
