# example
# http://matpalm.com/blog/2012/12/27/dead_simple_pymc/~

from numpy.random import normal
import random, numpy
from pymc.Matplot import plot
from pymc import *
from matplotlib import pyplot

data = [normal(100, 20) for _i in xrange(1000)]  # 2/3rds of the data
data += [normal(200, 20) for _i in xrange(500)]  # 1/3rd of the data
random.shuffle(data)
data[0] = 50
data[1] = 250
 
theta = Uniform("theta", lower=0, upper=1)
bern = Bernoulli("bern", p=theta, size=len(data))
# what does size do here?
 
mean1 = Uniform('mean1', lower=min(data), upper=max(data))
mean2 = Uniform('mean2', lower=min(data), upper=max(data))
std_dev = Uniform('std_dev', lower=0, upper=50)
 
@deterministic(plot=False)
def mean(bern=bern, mean1=mean1, mean2=mean2):
    return bern * mean1 + (1 - bern) * mean2
 
@deterministic(plot=False)
def precision(std_dev=std_dev):
    return 1.0 / (std_dev * std_dev)
 
process = Normal('process', mu=mean, tau=std_dev, value=data, observed=True)
model = Model([theta, bern, mean1, mean2, std_dev, mean, precision, process])
m = MCMC(model)
m.sample(iter=100000, burn=1000)
print(m.stats())

# for p in ['mean1', 'mean2', 'std_dev', 'theta']:
#     numpy.savetxt("%s.trace" % p, m.trace(p)[:])

# I don't understand what this does.
fig = pyplot.figure()
pyplot.title("Posterior distribution of mean1")
pyplot.hist(m.trace("mean1")[:], bins=25, histtype="stepfilled", normed=True)
pyplot.savefig('../plots/example1Mean1.png')
pyplot.close(fig)

fig = pyplot.figure()
pyplot.title("Posterior distribution of mean2")
pyplot.hist(m.trace("mean2")[:], bins=25, histtype="stepfilled", normed=True)
pyplot.savefig('../plots/example1Mean2.png')
pyplot.close(fig)

fig = pyplot.figure()
pyplot.title("Posterior distribution of theta")
print( m.trace("theta")[0:10] )
pyplot.hist(m.trace("theta")[:], bins=25, histtype="stepfilled", normed=True)
pyplot.savefig('../plots/example1Theta.png')
pyplot.close(fig)

for x in range(3):
    fig = pyplot.figure()
    pyplot.title("Posterior distribution of bern%d" % (x,))
    pyplot.hist([run[x] for run in m.trace("bern")[:]], bins=25,
                histtype="stepfilled", normed=True)
    pyplot.savefig('../plots/example1Bern%d.png' % (x,))
    pyplot.close(fig)
