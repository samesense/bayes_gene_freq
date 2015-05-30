import sys, pickle, pymc, numpy
from matplotlib import pyplot

def simEsp(espTraceFile, outPng):
    # 7 bases
    N = 7
    hyperPrior_mu = pymc.Uniform("esp_mu_hyper", 0, 1)
    hyperPrior_precision = pymc.Uniform("esp_precision_hyper", 0.001, 100)
    # values = numpy.array( 100*[False] + 30*[True]
    #                       + 50*[False] + 30*[True] )

    values = numpy.array( 100*[False] + 30*[True]
                          + 50*[False] + 30*[True]
                          + 2000*[False] + 1000*[True]
                          + 100*[False] + 40*[True]
                          + 100*[False] + 50*[True]
                          + 100*[False] + 60*[True]
                          + 1000*[False] + 500*[True] )
                          
    idx = [0]*130 + [1]*80 + [2]*3000 + [3]*140 \
          + [4]*150 + [5]*160 + [6]*1500
    
    target_freq = pymc.Normal('esp_freq', mu=hyperPrior_mu,
                              tau=hyperPrior_precision, size=N)
    observations = pymc.Bernoulli("esp_obs", target_freq[idx],
                                  observed=True,
                                  value=values)
    model = pymc.Model([hyperPrior_mu, hyperPrior_precision,
                        target_freq, observations])
    mcmc = pymc.MCMC(model)
    mcmc.sample(iter=100000, burn=15000)

    t = mcmc.trace("esp_mu_hyper")[:]
    fig = pyplot.figure()
    pyplot.title("Posterior distribution of esp freq mu")
    pyplot.hist(t, bins=25,
                histtype="stepfilled", normed=True)
    pyplot.savefig(outPng)
    pyplot.close(fig)

    with open(espTraceFile, 'w') as fout:
        pickle.dump(t, fout)

    fig = pyplot.figure()
    pyplot.title("Posterior distribution of esp each mu freq")
    for run in range(N):
        t = [ x[run] for x in mcmc.trace("esp_freq")[:] ]
        pyplot.subplot(4, 2, run+1)
        pyplot.hist(t, bins=25,
                    histtype="stepfilled", normed=True)
    pyplot.savefig('../plots/espF.png')
    pyplot.close(fig)
        
if __name__ == '__main__':
    espTraceFile, outPng = sys.argv[1:]
    simEsp(espTraceFile, outPng)
