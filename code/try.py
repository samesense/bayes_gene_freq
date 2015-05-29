import pymc, numpy
from matplotlib import pyplot

# single base
p = pymc.Uniform("freq_variant", 0, 1)
values = 100*[False] + 5*[True]
observations = pymc.Bernoulli("obs", p, observed=True, value=values)
model = pymc.Model([p, observations])
mcmc = pymc.MCMC(model)
mcmc.sample(40000, 15000)

pyplot.title("Posterior distribution of $p_A$, the true effectiveness of site A")
pyplot.hist(mcmc.trace("freq_variant")[:], bins=25, histtype="stepfilled", normed=True)
pyplot.savefig('../plots/test.png')

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

