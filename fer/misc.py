import matplotlib
import matplotlib.pyplot as plt
import sys
import hashlib
import os
import time

def setMatPlotLib(style='inline'):
    plt.plot([0, 1], [0, 1])
    plt.show()
    if style == 'inline':
        matplotlib.use('module://matplotlib_inline.backend_inline')
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    elif style == 'latex':
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            'pgf.texsystem': 'pdflatex',
            'font.family': 'serif',
            'text.usetex': True,
            "pgf.preamble": "\n".join([
                 "\\usepackage{amsmath}"
            ]),
            'pgf.rcfonts': False,
            'figure.dpi': 200,
            'legend.fontsize': 'small',
            'axes.labelsize': 'small',
            'axes.titlesize': 'small',
            'xtick.labelsize': 'small',
            'ytick.labelsize': 'small',
            'legend.markerscale': 0.6,
            'legend.handlelength': 1.0
        })
        
def logToFile(file):
    f = open(file, 'w')
    sys.stdout = f
    sys.stderr = f
    global print
    print = partial(print, flush=True)
    
    
def md5(x):
    return hashlib.md5(str(x).encode()).hexdigest()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def convertMillis(millis):
    seconds=int((millis/1000)%60)
    minutes=int((millis/(1000*60))%60)
    hours=int(millis/(1000*60*60))
    return f'{hours}:{minutes:02d}:{seconds:02d}'
    
def inclog(message, step, total, inittime):
    perc = step * 100 / total
    curtime = time.time()
    elapsed = (curtime - inittime) * 1000
    if perc > 0:
        remaining = elapsed / step * total
    else:
        remaining = 0
    elapsed = convertMillis(elapsed)
    remaining = convertMillis(remaining)
    print(f'{message} {perc:.0f}% [{step}/{total}] {elapsed} / {remaining}', flush=True)