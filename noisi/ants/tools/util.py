from obspy import read_inventory
from obspy.geodetics import gps2dist_azimuth
import re
import os

def get_geoinf(id1,id2):

		inv1 = '{}.{}.xml'.format(*id1.split('.')[0:2])
		inv2 = '{}.{}.xml'.format(*id2.split('.')[0:2])

		inv1 = read_inventory(os.path.join('meta','stationxml',inv1))
		inv2 = read_inventory(os.path.join('meta','stationxml',inv2))

		# Replace 'radial' and 'transverse' by 'N' and 'E'
		id1 = re.sub('\.??R$','N',id1)
		id2 = re.sub('\.??R$','N',id2)
		id1 = re.sub('\.??T$','E',id1)
		id2 = re.sub('\.??T$','E',id2)
		

		c1 = inv1.get_coordinates(id1)
		c2 = inv2.get_coordinates(id2)

		lat1, lon1, lat2, lon2 = (
			c1['latitude'],
			c1['longitude'],
			c2['latitude'],
			c2['longitude'])

		dist, az, baz = gps2dist_azimuth(lat1,lon1,lat2,lon2)

		return lat1, lon1, lat2, lon2, dist, az, baz