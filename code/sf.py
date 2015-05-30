"""Snakefile."""

rule traceTarget:
    output: '../working/target.trace',
            '../plots/target.png'
    shell: 'python traceTarget.py {output}'

rule traceEsp:
    output: '../working/esp.trace',
            '../plots/esp.png'
    shell: 'python traceEsp.py {output}'

rule cmpTraces:    
    input: '../working/target.trace', \
           '../working/esp.trace'
    output: '../plots/cmp.png'
    shell: 'python cmpTraces.py {input} {output}'
