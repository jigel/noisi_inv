import matplotlib
matplotlib.use('tkagg')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import mlab

import matplotlib.tri as tri
from scipy.signal import sosfilt
from glob import glob

import os
try:
    import h5py
except ImportError:
    print('h5py cannot be used and if you set output="ASDF"\
        or write_intermediate > 0, processing will fail.')
import time
import numpy as np

from obspy import read_inventory, read, Stream
from obspy.signal.invsim import cosine_taper
from ants_2.tools.treatment import bandpass as get_bandpass


class stainfo(object):

    def __init__(self,staid):

        self.id = staid
        self.lat = None
        self.lon = None

def plot_stations(projection='merc',data='processed',
    channels=['BHZ','LHZ'],locations = ['','00','10'],bluemarble=False):


    # figure out station IDS and their coordinates
    ids = []
    stations = []
    lats = []
    lons = []

    files = os.listdir(os.path.join('data',data))
    for f in files:
        infs = os.path.basename(f).split('.')
        if len(infs) < 4: continue
        ids.append('{}.{}'.format(*infs[0:2]))
    ids = list(set(ids))

    if ids == []:
        print('No data found.')
        return()
    
    # look up station xml files, get coordinates
    for i in ids:
        station = stainfo(i)
        
        staxml = os.path.join('meta','stationxml',i+'.xml')
        if os.path.exists(staxml) == False:
            continue

        inv = read_inventory(staxml)

        for j in range(len(locations)):
            for k in range(len(channels)):

                if station.lat is not None: break

                try:
                    c = inv.get_coordinates(i+'.'+locations[j]+
                        '.'+channels[k])
                    station.lat = c['latitude']
                    station.lon = c['longitude']
                    lats.append(c['latitude'])
                    lons.append(c['longitude'])
                except:
                    continue
            
        if station.lat == None:
            print('No coordinates found for station %s\
for locations \'\',00,10.' %i)
            continue
        else:
            stations.append(station)
    
    fig = plt.figure()


    # if min(lons) < 0:
    #   lons += 180.
    #   shift_lons = True
    # if min(lats) < 0:
    #   lats += 90.
    #   shift_lats = True

    # xmin, xmax, ymin, ymax and central meridian of map
    xmin = min(lons) - 10 
    if xmin < -180:
        xmin = -180.
    xmax = max(lons) + 10
    if xmax > 180.:
        xmax = 180.
    ymin = min(lats) - 5
    if ymin < -90.:
        ymin = -90.
    ymax = max(lats) + 5
    if ymax > 90.:
        ymax = 90.


    mid_merid = (max(lons) - min(lons)) * 0.5
    mid_paral = (max(lats) - min(lats)) * 0.5
    

    # basemap
    m = Basemap(rsphere=6378137,
                resolution='l',
                projection=projection,
                lat_0=mid_paral,
                lon_0=mid_merid,
                llcrnrlat=ymin,
                urcrnrlat=ymax,
                llcrnrlon=xmin,
                urcrnrlon=xmax)

    
    if bluemarble:
        m.bluemarble()
        textcol = 'w'
    else:
        m.drawcoastlines()
        textcol = 'k'

    #draw the meridians and parallels
    parallels = np.arange(round(ymin),round(ymax),10)
    #labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(round(xmin),round(xmax),20)
    m.drawmeridians(meridians,labels=[True,False,False,True])

    # plot stations on map
    for sta in stations:
        

        m.plot(sta.lon,sta.lat,'rv',markersize=12.,latlon=True)
        x, y = m(sta.lon,sta.lat)
        plt.text(x,y,'   '+sta.id,fontweight='bold',color=textcol)
    # save map in test folder
    plt.show()


def plot_converging_stack(inputfile,bandpass=None,pause=0.):

    f = h5py.File(inputfile,'r')
    plt.ion()

    stack = list(f['corr_windows'].keys())[0]
    stack = f['corr_windows'][stack][:]
    stats = f['stats']
    Fs = stats.attrs['sampling_rate']
    cha1 = stats.attrs['channel1']
    cha2 = stats.attrs['channel2']

    if bandpass is not None:
        taper = cosine_taper(len(stack))
        stack *= taper
        sos = get_bandpass(df=Fs,freqmin=bandpass[0],
            freqmax=bandpass[1],
            corners=bandpass[2])
        firstpass = sosfilt(sos, stack)
        stack =  sosfilt(sos, firstpass[::-1])[::-1]


    # display a counter for stacked windows
    cnt = 1

    
    max_lag = ((len(stack) - 1) / 2) / Fs
    lag = np.linspace(-max_lag,max_lag,len(stack))

    fig = plt.figure()
    ax1 = fig.add_subplot(212)

    ax1.set_title('{}--{}'.format(cha1,cha2))
    ax1.set_xlabel('Lag (s)')
    ax1.set_ylabel('Correlation stack')
    line1, = ax1.plot(lag,stack,'k')
    

    ax2 = fig.add_subplot(211)
    ax2.set_ylim([np.min(stack)*3,np.max(stack)*3])
    ax2.set_ylabel('Correlation window(s)')
    line2, = ax2.plot(lag,stack)
    text1 = ax2.set_title(str(cnt))
    print(matplotlib.get_backend())
    

    fig.canvas.draw()
    plt.show(block=False)

    for key in list(f['corr_windows'].keys()):

        cwindow = f['corr_windows'][key][:]
        try:
            rms1 = f['corr_windows'][key].attrs['rms1']
            rms2 = f['corr_windows'][key].attrs['rms2']
        
            if rms1.max() > 1.e-08 or rms2.max() > 1.e-08:
                continue
        except:
            pass

        if bandpass is not None:
            
            cwindow *= taper
            firstpass = sosfilt(sos, cwindow)
            cwindow =  sosfilt(sos, firstpass[::-1])[::-1]

        stack += cwindow
        
        ax1.set_ylim([np.min(stack)*1.5,np.max(stack)*1.5])
        text1.set_text(str(cnt))

        
        

        line1.set_ydata(stack)
        line2.set_ydata(cwindow)
        

        fig.canvas.draw()
        fig.show()
        time.sleep(0.01)
        #fig.canvas.draw()
        cnt += 1
        if pause > 0:
            time.sleep(pause)




def plot_correlation(f, bandpass=None):

    tr = read(f)[0]

    if bandpass is not None:
        tr.filter('bandpass',freqmin=bandpass[0],
            freqmax=bandpass[1],corners=bandpass[2])

    n_stack = tr.stats.sac['user0']

    maxlag = (tr.stats.npts-1) / 2 * tr.stats.delta
    lag = np.linspace(-maxlag,maxlag,tr.stats.npts)

    plt.plot(lag,tr.data / n_stack)

    id2 = '{}.{}.{}.{}'.format(tr.stats.sac.kuser0.strip(),
        tr.stats.sac.kevnm.strip(),
        tr.stats.sac.kuser1.strip(),
        tr.stats.sac.kuser2.strip())

    trid = tr.id + '--' + id2
    plt.title(trid)
    plt.grid()
    plt.show()



def plot_section(pathname,bandpass=None,fmt='SAC'):

    
    inpt = glob(os.path.join(pathname,'*.{}'.format(fmt.lower())))
    inpt.extend(glob(os.path.join(pathname,'*.{}'.format(fmt.upper()))))

    traces = Stream()
    for path in inpt:
        try:
            traces += read(path)[0]
        except:
            continue


    for t in traces:

        t.stats.distance = t.stats.sac.dist

    if bandpass is not None:
        traces.filter('bandpass',freqmin=bandpass[0],
            freqmax=bandpass[1],corners=bandpass[2])

    # maxlag seems to be requested in samples..this must be a bug in obspy.
    maxlag = (traces[0].stats.npts-1) / 2.0 
    
    traces.plot(type='section',orientation='horizontal',
        reftime = traces[0].stats.starttime + maxlag)




def plot_window(correlation, window, measurement,win_noise=None):
    
    
    maxlag = (correlation.stats.npts - 1) / 2. * correlation.stats.delta
    lag = np.linspace(-maxlag,maxlag,correlation.stats.npts)
    
    plt.plot(lag,correlation.data/np.max(np.abs(correlation.data)))
    plt.plot(lag,window/np.max(np.abs(window)),'--',linewidth=1.5)
    if win_noise is not None:
        plt.plot(lag,win_noise,'--',linewidth=1.5)

    plt.legend(['data','signal window','noise window'])
    plt.title(correlation.id)
    plt.text(0,-0.75,'Measurement value: %g' %measurement)
    plt.xlabel('Correlation Lag in seconds.')
    plt.ylabel('Normalized correlation and window(s).')
    
    plt.show()



def plot_grid(map_x,map_y,map_z,stations=[],vmin=-1.2,
    vmax=1.2,outfile=None,title=None,shade='flat',cmap='div'):


    
    lonmin = np.min(map_x)
    lonmax = np.max(map_x)
    latmax = np.max(map_y)
    latmin = np.min(map_y)

    if cmap == 'seq':
        cmap = plt.cm.BuGn

    elif cmap == 'div':
        cmap = plt.cm.bwr

    m = Basemap(rsphere=6378137,resolution='c',projection='cyl',
    llcrnrlat=latmin,urcrnrlat=latmax,
    llcrnrlon=lonmin,urcrnrlon=lonmax)

    
    triangles = tri.Triangulation(map_x,map_y)

    # tripcolor plot.
    plt.figure()
    plt.subplot(111)
    plt.gca().set_aspect('equal')

    if title is not None:
        plt.title(title)

    plt.tripcolor(triangles, map_z/np.max(np.abs(map_z)),
        shading=shade, vmin=vmin,vmax=vmax, cmap=cmap)


    m.colorbar(location='bottom',pad=0.4)
    m.drawcoastlines(linewidth=2.0)
    d_lon = round(abs(lonmax-lonmin) / 5.)
    d_lat = round(abs(latmax-latmin) / 5.)
    parallels = np.arange(latmin,latmax,d_lat).astype(int)
    meridians = np.arange(lonmin,lonmax,d_lon).astype(int)
    m.drawparallels(parallels,labels=[1,0,0,0]) # draw parallels
    m.drawmeridians(meridians,labels=[0,0,0,1])
    

    #draw station locations
    for sta in stations:
        m.plot(sta[0],sta[1],'rv',latlon=True)

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile,format='png')


def plot_spectrogram(trace,win_samples,overlap,pad=None,mode='psd',
    fmin=None,fmax=None,clip=None,cmap=plt.cm.plasma,figshape=(10,2),dpi=200, 
    time_divide=86400,title=None,outfile=None):
    
    spec, freq, t = mlab.specgram(trace.data, NFFT=win_samples, Fs=trace.stats.sampling_rate,
        detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=overlap,
        pad_to=pad, sides='default',scale_by_freq=None, mode=mode)
    
    
    
    plt.figure(figsize=figshape,dpi=dpi)
    ax = plt.subplot(111)
    
    if fmin is not None:
        i0 = np.argmin(np.abs(freq-fmin))
    else:
        i0 = 0
    if fmax is not None:
        i1 = np.argmin(np.abs(freq-fmax))
    else:
        i1 = -1
    
    
    
    
    
    if clip is not None:
        vmin = clip[0]
        vmax = clip[1]
    else:
        vmin = np.min(spec)
        vmax = np.max(spec)


    spec_plot = np.flipud(spec[i0:i1,:])

    ax.imshow(spec_plot,
        extent=(t[0]/time_divide,t[-1]/time_divide,freq[i0],freq[i1]),
        vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest')
    
    ax.axis('tight')
    if title is None:
        plt.title(trace.stats.starttime)
    else:
        plt.title(title)

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
        plt.close('all')
