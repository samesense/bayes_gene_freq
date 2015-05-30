import sys, pickle
from matplotlib import pyplot

def runCmp(t, e, outPng):
    fig = pyplot.figure()
    pyplot.title("Posterior distribution of target freq")
    pyplot.hist(t-e, bins=25,
                histtype="stepfilled", normed=True)
    pyplot.savefig(outPng)
    pyplot.close(fig)

if __name__ == '__main__':
    targetTraceFile, espTraceFile, outPng = sys.argv[1:]
    with open(targetTraceFile) as f:
        t = pickle.load(f)
    with open(espTraceFile) as f:
        e = pickle.load(f)
    runCmp(t, e, outPng)
