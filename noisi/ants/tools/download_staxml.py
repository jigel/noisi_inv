"""
This code is from the ants package: https://github.com/lermert/ants_2.

:copyright:
    noisi development team
:license:
    GNU Lesser General Public License, Version 3 and later
    (https://www.gnu.org/copyleft/lesser.html)

"""

from obspy.clients import fdsn
import sys
import os

stationlist = open(sys.argv[1]).read().split('\n')
outdir = sys.argv[2]

client = fdsn.Client()

for station in stationlist:
    if station == '':
        continue
    try:
        net = station.split()[0].upper()
        sta = station.split()[1].upper()
    except IndexError:
        net = station.split('.')[0].upper()
        sta = station.split('.')[1].upper()
    if net=='':
        continue
    xml = '%s.%s.xml' %(net,sta)
    xml = os.path.join(outdir,xml)
    try:
        client.get_stations(network=net,station=sta,
            filename=xml,level='response')
    except:
        continue
