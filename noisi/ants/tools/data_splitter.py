from obspy import read, UTCDateTime
from glob import glob
import os
####################
# user input
input_directory = '/Volumes/Japan_sea/temp/'#'/mnt/lnec/lermert/hum_reprocessing/data/raw/temp/'
# start time
start = UTCDateTime('2004,01,01')
endtime = UTCDateTime('2005,01,01')
# interval in seconds
step = 30*86400
# output directory
output_directory = '/Volumes/Japan_sea/temp_out/'#'/mnt/lnec/lermert/hum_reprocessing/data/raw/'
####################


# get a list of the files
files = glob(os.path.join(input_directory,'*.mseed'))
print(start)
print(endtime-step)
print(files)

# loop over files
for f in files:
	# read file
	tr = read(f)
	for t in tr:
		t.stats.sampling_rate = round(t.stats.sampling_rate,4)

	tr.merge(method=1,interpolation_samples=0)
	

	print(tr)
	t = start
	# loop over time windows
	while t < (endtime-step):
		
		# split it
		trtemp = tr.slice(starttime=t,endtime=t+step)
		print(trtemp)
		s1 = t.strftime("%Y.%j.%H.%M.%S")
		s2 = (t+step).strftime("%Y.%j.%H.%M.%S")


		
		for trc in trtemp:
			newstr = trc.split()
			
			for tc in newstr:
				print(tc)
				filename = '{}.{}.{}.mseed'.format(tc.id,s1,s2)
				filename = os.path.join(output_directory,filename)
				if tc.stats.npts > 0:
				# save it
					tc.write(filename,format='MSEED')
		t += step
