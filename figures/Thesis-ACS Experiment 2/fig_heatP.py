from stat import S_ISREG, ST_CTIME, ST_MODE, ST_MTIME
import os, sys, time, re, math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
def loadCSV(path):
    return np.loadtxt(
         path,
         dtype=[('bins','int'),
                ('pop','int'),
                ('actors per bin','float'), ('original overall','float'), ('atomic overall','float')],
         skiprows=2,
         delimiter=',',
         usecols=(  2,
                    3,
                    4, 5, 11),
         unpack=True
     );
###        
### Locate the suitable files in the directory (.csv's)
###
pattern = re.compile("\.csv$");
# get all entries in the directory w/ stats
entries = (os.path.join('.', fn) for fn in os.listdir('.'))

# leave only regular files
entries = ((path)
           for path in entries if (bool (pattern.search(path))))

for path in entries:
    plt.clf();#Clear the entire figure
    plt.rc('font', family='serif', serif='Times')
    #plt.rc('text', usetex=True)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["legend.fontsize"] = 8
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(3.5, 3.5/1.4)
    #fig.set_size_inches(3.5*3, 3.5*3/1.4)
    #Label axis
    ax1.set_xlabel(r'Actor Count ($10^{6}$)');
    ax1.set_ylabel('Avg Actors Per Bin');
    #ax2 = ax1.twinx();
    #ax2.set_ylabel('Percentage (%)');
    ###
    ### Load Data, Create tuples of matching columns from each file
    ###
    csv = loadCSV(path);
    atomic = csv.pop(-1);
    original = csv.pop(-1);
    actorsPerBin = csv.pop(-1);
    actorPop = csv.pop(-1);
    binCount = csv.pop(-1);
    ###
    ### Filter data
    ###
    colorData = [];
    actorPop2 = [];
    for i in range(len(actorPop)):
        colorData.append((atomic[i]/original[i])*100); #MS  
        actorPop2.append(actorPop[i]/1000000.0);
    ###
    ### Count items
    ###
    actorItems = set();#should come out 15 (16-1)
    apbItems = 0;#should come out 18  (19-1)  
    for i in range(1, len(actorPop)):
        if actorPop[i]==actorPop[0]:
            apbItems = apbItems+1;
        actorItems.add(actorPop[i]);
    actorItems = len(actorItems)-1;
    ###
    ### PlotGraph
    ###
    plt.hexbin(actorPop2, actorsPerBin, C=colorData, gridsize=(actorItems,apbItems), cmap=cm.jet, bins=None)
    plt.axis([min(actorPop2), max(actorPop2), min(actorsPerBin), max(actorsPerBin)])            
    ###
    ### Position Legend
    ###
    cb = plt.colorbar()
    cb.set_label('ACS/CUBWord (%)')
    plt.tight_layout();
    ###
    ### Extract name, sans filetype
    ###
    fileName = os.path.splitext(os.path.basename(path))[0]
    ###
    ### Export/Show Plot
    ###
    #plt.savefig('[HeatP]'+ fileName + '.pdf')
    plt.savefig('[HeatP]'+ fileName + '.pdf')
    #plt.savefig('[HeatP]'+ fileName + '.pdf')
    plt.close();
    #plt.show();
