"""Snakefile."""

WORK = '../working/'
PLOTS = '../plots/'

rule clean:
    run:
        for afile in (WORK + 'target.trace', \
                      PLOTS + 'target.png', \
                      WORK + 'esp.trace', \
                      PLOTS + 'esp.png', \
                      PLOTS + 'cmp.png'):
            shell('rm ' + afile)

rule traceTarget:
    output: WORK + 'target.trace', \
            PLOTS + 'target.png'
    shell: 'python traceTarget.py {output}'

rule traceEsp:
    output: WORK + 'esp.trace', \
            PLOTS + 'esp.png'
    shell: 'python traceEsp.py {output}'

rule cmpTraces:    
    input: WORK + 'target.trace', \
           WORK + 'esp.trace'
    output: PLOTS + 'cmp.png'
    shell: 'python cmpTraces.py {input} {output}'
