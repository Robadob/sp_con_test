from stat import S_ISREG, ST_CTIME, ST_MODE, ST_MTIME
import os, sys, time, re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
  
def plotLine(name, xData, yData, color, symbol, line='-', doPoints=True):
    if len(xData)!=len(yData):
        print("Len x and y do not match: %d vs %d" %(len(xData), len(yData)));
    #Sort data
    xData, yData = zip(*sorted(zip(xData, yData)))
    #Array of sampling vals for polyfit line
    xp = np.linspace(xData[0], xData[-1]*0.98, 100)
    #polyfit
    #default_z = np.polyfit(xData, yData, 6)
    #default_fit = np.poly1d(default_z)
    # plt.plot(
        # xp, 
        # default_fit(xp), 
        # str(color)+str(line),
        # label=str(name),
        # lw=1
    # );
    #points
    if(doPoints):
        default_h = plt.plot(
           xData,yData, 
           str(symbol),
           label=str(name),
           lw=1,
           color=color
        );
        
def loadCSV(path):
    return np.loadtxt(
         path,
         dtype=[('Bin Count','int'), ('Population','int'),('atomic overall(ms)','float'),('atomic memset(ms)','float'),('atomic histogram(ms)','float'),('atomic scan(ms)','float'),('atomic reorder(ms)','float'),('atomic tex(ms)','float')],
         skiprows=2,
         delimiter=',',
         usecols=(2,3,10,11,12,13,14,15),
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

###
### Config, labelling
###
colors = ['#000000','#000000','#000000','#999999','#999999','#999999'];
symbols = ['*', 'o', '^', 'x', 's', '+', 'h','p'];
lines = ['-','--',':', '-', '--', ':'];
for path in entries:
    plt.clf();#Clear the entire figure
    plt.rc('font', family='serif', serif='Times')
    #plt.rc('text', usetex=True)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["legend.fontsize"] = 8
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    fig = plt.figure()
    fig.set_size_inches(3.5, 3.5/1.4)
    #fig.set_size_inches(3.5*3, 3.5*3/1.4)
    co = 6;#5: overall step time, #6: kernel time, #7: rebuild/texture time
    #Label axis
    plt.xlabel('Bin Count');
    plt.ylabel('Overall Construction Time (ms)');
    plt.ticklabel_format(style='sci', useMathText=True, axis='x', scilimits=(0,0));
    ###
    ### Load Data, Create tuples of matching columns from each file
    ###
    csv = loadCSV(path);
    atomic_tex = csv.pop(-1);
    atomic_reorder = csv.pop(-1);
    atomic_scan = csv.pop(-1);
    atomic_histogram = csv.pop(-1);
    atomic_memset = csv.pop(-1);
    atomic_overall = csv.pop(-1);
    pop_size = csv.pop(-1);
    bin_count = csv.pop(-1);
    ###
    ### Filter data, Only publish bin width's that we want
    ###
    plotLine('Overall', bin_count, atomic_overall, colors[0], lines[0])
    plotLine('Memset', bin_count, atomic_memset, colors[1], lines[1])
    plotLine('Histogram', bin_count, atomic_histogram, colors[2], lines[2])
    plotLine('Scan', bin_count, atomic_scan, colors[3], lines[3])
    plotLine('Reorder', bin_count, atomic_reorder, colors[4], lines[4])
    plotLine('Tex', bin_count, atomic_tex, colors[5], lines[5])
    ###
    ### Position Legend
    ###
    #plt.legend(loc='lower right',numpoints=1);
    plt.legend(loc='best',numpoints=1);
    plt.tight_layout();
    ###
    ### Extract name, sans filetype
    ###
    fileName = os.path.splitext(os.path.basename(path))[0]
    ###
    ### Export/Show Plot
    ###
    #plt.savefig('[Atomic]'+ fileName + '.pdf')
    plt.savefig('[Atomic]'+ fileName + '.pdf')
    #plt.savefig('[Atomic]'+ fileName + '.pdf')
    plt.close();
    #plt.show();
