from stat import S_ISREG, ST_CTIME, ST_MODE, ST_MTIME
import os, sys, time, re, math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
    
csType = 8;#7;;#8== filter for actors per bin, 7 == filter for pop
filterArg = 99.4391;#846666;
path = "[TitanV]sweep-sorted1561374035.csv"
  
def plotLine(axis, name, xData, yData, color, symbol, line='-', doPoints=True):
    if len(xData)!=len(yData):
        print("Len x and y do not match: %d vs %d" %(len(xData), len(yData)));
    #Sort data
    xData, yData = zip(*sorted(zip(xData, yData)))
    #Array of sampling vals for polyfit line
    #xp = np.linspace(xData[0], xData[-1]*0.98, 100)
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
        default_h = axis.plot(
           xData,yData, 
           str(symbol),
           label=str(name),
           lw=1,
           color=color
        ); 
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
### Config, labelling
###
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#dc5939', '#e8b2d8' ,'#684204' ,'#d9ffa0' ,'#a293bf'];
symbols = ['*', 'o', '^', 'x', 's', '+', 'h','p'];
lines = ['-','--',':', '-.']
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)#Can't use this, some latex font cache file missing
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.fontsize"] = 8
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
#plt.rc('text', usetex=True)
fig, ax1 = plt.subplots()
fig.set_size_inches(3.5, 3.5/1.4)
#fig.set_size_inches(3.5*3, 3.5*3/1.4)
#Label axis

if csType==7:
    ax1.set_xlabel('Avg Actors Per Bin');
elif csType==8:
    ax1.set_xlabel(r'Actor Count ($10^{6}$)');
ax1.set_ylabel('Construction Time (ms)');
ax1.ticklabel_format(style='sci', useMathText=True, axis='x', scilimits=(0,0))
ax2 = ax1.twinx();
ax2.set_ylabel('Percentage (%)');

###
### Load Data, Create tuples of matching columns from each file
###
csv = loadCSV(path);
atomic = csv.pop(-1);
original = csv.pop(-1);
actorsPerBin = csv.pop(-1);
actorPop = csv.pop(-1);
binCount = csv.pop(-1);
xVals = [];
yAtomic = [];
yOriginal = [];
colorData = [];
###
### Filter data
###
for i in range(len(actorPop)):
    if csType==7:#Filter agent count
        if actorPop[i]==filterArg:
            xVals.append(actorsPerBin[i]);
            yAtomic.append(atomic[i]);
            yOriginal.append(original[i]);
            colorData.append((atomic[i]/original[i])*100);
    elif csType==8:#Filter neighbourAvg
        if abs(actorsPerBin[i]-filterArg)<0.01:#Epsilon equal within range 0.1
            xVals.append(actorPop[i]/1000000.0);
            yAtomic.append(atomic[i]);
            yOriginal.append(original[i]);
            colorData.append((atomic[i]/original[i])*100);
if len(xVals) == 0:
    print("Filter found no results.");
    sys.exit();
###
### Plot Graph
###  
plotLine(ax1, "Original", xVals, yOriginal, '#000000', ':');
plotLine(ax1, "ACS", xVals, yAtomic, '#000000', '--');
plotLine(ax2, "ACS/Original", xVals, colorData, '#999999', '-');       
###
### Position Legend
###
ax1.legend(loc='upper left',numpoints=1);
ax2.legend(loc='lower right',numpoints=1);
plt.tight_layout();
###
### Extract name, sans filetype
###
fileName = os.path.splitext(os.path.basename(path))[0]
###
### Export/Show Plot
###
if csType==7:
    plt.savefig('%s(Population=%d).pdf' % (fileName, filterArg))
elif csType==8:
    plt.savefig('%s(Density=%.2f).pdf' % (fileName, filterArg))
plt.close();
#plt.show();
