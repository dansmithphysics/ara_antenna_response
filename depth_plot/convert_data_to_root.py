import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import ROOT as root

f = open("./drop_data.csv")

list = f.readlines()
lines = list[0].split("\r")

index_remove = [i for i in range(len(lines)) if lines[i] == '']
lines = np.delete(lines, index_remove)

times  = [lines[i].split(",")[0] for i in range(1,len(lines))]
depths = [lines[i].split(",")[1] for i in range(1,len(lines))]
times_unix = []

for i in range(len(times)):
    print times[i], depths[i]
    h, m, s = times[i].split(":")
    h, m, s = int(h), int(m), int(s)
    d = 10
    if(h > 24):
        h -= 24
        d += 1
    elif(h == 13):
        h = 15
    dt = datetime.datetime(2018, 1, d, h, m, s)
    offset = -13 * 60 * 60 - 6 * 60 * 60

    times_unix += [int(time.mktime(dt.timetuple()) + offset)]

#for i in range(len(times_unix)):
#    print times_unix[i], depths[i]
#plt.plot(times_unix, depths)
#plt.xlabel("Times, Unix [s]")
#plt.ylabel("Depth [m]")
#plt.show()

g = open("./unix_corrected_drop_data.csv", "w")
for i in range(len(times_unix)):
    g.write(str(times_unix[i])+","+str(times[i])+","+str(depths[i])+"\n")
g.close()

depth_vs_time = root.TGraph(len(times_unix))
for i in range(len(times_unix)):
    depth_vs_time.GetX()[i] = times_unix[i] - 1515549411;
    depth_vs_time.GetY()[i] = float(depths[i])

nf = root.TFile("depth_vs_time.root", "recreate")
depth_vs_time.Write("depth_vs_time")
nf.Close()
