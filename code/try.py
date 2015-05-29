import pymc, numpy
from matplotlib import pyplot

def runSingleBase():
    # single base
    p = pymc.Uniform("freq_variant", 0, 1)
    values = 100*[False] + 5*[True]
    observations = pymc.Bernoulli("obs", p, observed=True, value=values)
    model = pymc.Model([p, observations])
    mcmc = pymc.MCMC(model)
    mcmc.sample(40000, 15000)

    pyplot.title("Posterior distribution of true effectiveness of site A")
    pyplot.hist(mcmc.trace("freq_variant")[:], bins=25,
                histtype="stepfilled", normed=True)
    pyplot.savefig('../plots/testSingle.png')

def runMultipleBases():
    # 7 bases
    N = 7
    hyperPrior_mu = pymc.Uniform("target_mu_hyper", 0, 1)
    hyperPrior_precision = pymc.Uniform("target_precision_hyper", 0.0001, 100)
    values = numpy.array( 100*[False] + 5*[True]
                          + 50*[False] + 5*[True]
                          + 2000*[False] + 5*[True]
                          + 100*[False] + 7*[True]
                          + 100*[False] + 8*[True]
                          + 100*[False] + 10*[True]
                          + 1000*[False] + 90*[True]
                      )
    idx = [0]*105 + [1]*55 + [2]*2005 + [3]*107 + [4]*108 + [5]*110 + [6]*1090
    target_freq = pymc.Normal('target_freq', mu=hyperPrior_mu,
                              tau=hyperPrior_precision, size=N)
    observations = pymc.Bernoulli("obs", target_freq[idx], observed=True,
                                  value=values[idx])
    model = pymc.Model([hyperPrior_mu, hyperPrior_precision,
                        target_freq, observations])
    mcmc = pymc.MCMC(model)
    mcmc.sample(iter=100000, burn=1000) #sample(40000, 15000)

    pyplot.title("Posterior distribution of target freq")
    pyplot.hist(mcmc.trace("target_mu_hyper")[:], bins=25,
                histtype="stepfilled", normed=True)
    pyplot.savefig('../plots/testMult.png')

# def runFull():
#     # 3 bases
#     N = 3
#     p_target = pymc.Uniform("freq_variant_target", 0, 1, size=N)
#     p_esp = pymc.Uniform("freq_variant_esp", 0, 1, size=N)
#     values_target = [100*[False] + 5*[True],
#                      50*[False] + 5*[True],
#                      2000*[False] + 5*[True]]
#     values_esp = [100*[False] + 50*[True],
#                   50*[False] + 50*[True],
#                   2000*[False] + 50*[True]]
#     observations_target = pymc.Bernoulli("obs", p_target, observed=True,
#                                          value=values_target, size=N)
#     observations_esp = pymc.Bernoulli("obs", p_esp, observed=True,
#                                          value=values_esp, size=N)
#     model = pymc.Model([p, observations])
#     mcmc = pymc.MCMC(model)
#     mcmc.sample(40000, 15000)

    # pyplot.title("Posterior distribution of $p_A$, the true effectiveness of site A")
    # pyplot.hist(mcmc.trace("freq_variant")[:], bins=25, histtype="stepfilled", normed=True)
    # pyplot.savefig('../plots/testMult.png')

runMultipleBases()

# N = 3
# target = pymc.Bernoulli("target", 0.01, size=N)
# esp = pymc.Bernoulli("esp", 0.01, size=N)

# #pymc.deterministic
# def summary(t=target, e=esp):
#     return numpy.average(t-e)

# p = pymc.Normal('avgDiff', mu=0, tau=1000)
# observations = [100*]
# model = pymc.Model([p, true_answers, target, esp, summary, observations])
# mcmc = pymc.MCMC(model)
# mcmc.sample(40000, 15000)

# X = 35
# N = 3
# observed_proportion = .5
# observations = pymc.Binomial("obs", N, observed_proportion, observed=True,
#                            value=X)
# print(observations)

