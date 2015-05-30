import sys, pickle, pymc, numpy
from matplotlib import pyplot

def simTarget(outFile, outPng):
    # 7 bases
    N = 7
    hyperPrior_mu = pymc.Uniform("target_mu_hyper", 0, 1)
    hyperPrior_precision = pymc.Uniform("target_precision_hyper",
                                        0.0001, 100)
    values = numpy.array( 100*[False] + 5*[True]
                          + 50*[False] + 5*[True]
                          + 2000*[False] + 5*[True]
                          + 100*[False] + 7*[True]
                          + 100*[False] + 8*[True]
                          + 100*[False] + 10*[True]
                          + 1000*[False] + 90*[True] )
    idx = [0]*105 + [1]*55 + [2]*2005 + [3]*107 \
          + [4]*108 + [5]*110 + [6]*1090
    target_freq = pymc.Normal('target_freq', mu=hyperPrior_mu,
                              tau=hyperPrior_precision, size=N)
    observations = pymc.Bernoulli("obs", target_freq[idx], observed=True,
                                  value=values)
    model = pymc.Model([hyperPrior_mu, hyperPrior_precision,
                        target_freq, observations])
    mcmc = pymc.MCMC(model)
    mcmc.sample(iter=100000, burn=15000)

    t = mcmc.trace("target_mu_hyper")[:]

    fig = pyplot.figure()
    pyplot.title("Posterior distribution of target freq mu")
    pyplot.hist(t, bins=25,
                histtype="stepfilled", normed=True)
    pyplot.savefig(outPng)
    pyplot.close(fig)

    with open(outFile, 'w') as fout:
        pickle.dump(t, fout)
        
if __name__ == '__main__':
    targetTraceFile, outPng = sys.argv[1:]
    simTarget(targetTraceFile, outPng)
