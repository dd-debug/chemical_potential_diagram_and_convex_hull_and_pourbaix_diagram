# encoding: utf-8
'''
Created on 2022.02.19

@author: jiadongc
'''
# V water 17.19 ice 1.92  cm3/mol
# S water -2.558, ice -23.55 J/mol/K

# G = -ST+VP+C
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
from scipy.spatial.qhull import ConvexHull
G0 = 38.619 # J/mol
Sw = -2.558
Vw = 17.19
Si = -23.55
Vi = 19.39
T0 = 273 #K
P0 = 21.643 #bar

Ci = G0+Si*T0-Vi*P0
Cw = G0+Sw*T0-Vw*P0
def calig(t, p):
    return Ci - Si*t+Vi*p
def calwg(t, p):
    return Cw - Sw*t+Vw*p
def calpco(t):
    #return p
    return ((Si-Sw)*t-(Ci-Cw))/(Vi-Vw)
def calpcop(p):
    #return T
    return ((Vi-Vw)*p+(Ci-Cw))/(Si-Sw)
print(calpcop(15), calpcop(35))

lowlimit = -500
vers1 = [[270,15,calig(270,15)],[270, 35, calig(270, 35)],
       [273.7, calpco(273.7), calig(273.7, calpco(273.7))], 
       [271.6, calpco(271.6), calig(271.6, calpco(271.6))]]
vers2 = [[275,15,calwg(275,15)],[275, 35, calwg(275, 35)],
       [273.7, calpco(273.7), calig(273.7, calpco(273.7))], 
       [271.6, calpco(271.6), calig(271.6, calpco(271.6))]]
vers1 = np.array(vers1)
vers2 = np.array(vers2)
print(vers1)
print(vers2)
fig = plt.figure(figsize=(9.2, 7))
ax = a3.Axes3D(fig)
ax.grid(b=None)
ax.dist=10
ax.azim=30
ax.elev=10
vers1_low = []
for ver in vers1:
    vercopy = ver.copy()
    vercopy[-1] = lowlimit
    vers1_low.append(vercopy)
vers1_low = np.array(vers1_low)
for vers in [vers1, vers1_low]:
    hull = ConvexHull(vers, qhull_options = "QJ")
    simplices = hull.simplices 
    # for s in simplices:
    #     print(s)
    #     print(vers1[s])
    org_triangles = [vers[s] for s in simplices]
    
    pc = a3.art3d.Poly3DCollection(org_triangles, \
         alpha = 0.03, facecolor="blue",edgecolor = None)
    ax.add_collection3d(pc)
# ax.scatter(vers1[:,0],vers1[:,1],vers1[:,2])

vers2_low = []
for ver in vers2:
    vercopy = ver.copy()
    vercopy[-1] = lowlimit
    vers2_low.append(vercopy)
vers2_low = np.array(vers2_low)
for vers in [vers2, vers2_low]:
    hull = ConvexHull(vers, qhull_options = "QJ")
    simplices = hull.simplices 
    # for s in simplices:
    #     print(s)
    #     print(vers1[s])
    org_triangles = [vers[s] for s in simplices]
    
    pc = a3.art3d.Poly3DCollection(org_triangles, \
         alpha = 0.03, facecolor="red",edgecolor = None)
    ax.add_collection3d(pc)

ax.plot([calpcop(15), calpcop(35)],[15, 35],[lowlimit,lowlimit], c = "k",alpha = 0.5)
ax.plot([calpcop(15), calpcop(15)],[15, 15],[lowlimit,calig(273.7, calpco(273.7))], 
        c = "k",alpha = 0.2, linestyle = "--")
ax.plot([calpcop(35), calpcop(35)],[35, 35],[lowlimit,calig(271.6, calpco(271.6))], 
        c = "k",alpha = 0.2, linestyle = "--")
ax.set_xlim([270,275])
ax.set_ylim([15,35])
ax.set_yticks(range(15,36,5))
ax.set_zlim([lowlimit, 300])
ax.set_zticks(np.arange(-200, 305, 100))
ax.dist=10
ax.azim=-135
ax.elev=18
plt.show()















