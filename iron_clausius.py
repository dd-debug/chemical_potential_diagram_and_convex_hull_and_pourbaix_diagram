'''
Created on 2022.02.19

@author: jdche
'''
# 鈭嗮潙�=鈭�0.07cm3/mol  alpha->gamma
# 鈭哠=0.69 J/(K mol) alpha->gamma
# 鈭嗮潙�=1.5鈭�2.2=鈭�0.7 (饾渿_饾惖) alpha->gamma
'''鈭嗮潙� need to normalized to J/mol, so  鈭嗮潙�= 鈭嗮潙�*6.02*0.927 ?'''
# initial condition: 1183K, 1 bar, 45 饾渿T
# plane function: T=  鈭嗮潙�/鈭哠*P - 鈭哅/鈭哠*H + constant

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


dV=-0.07
dS=0.69
dM=-0.7*6.02*0.927
print("dM",dM)
T = 1183
P = 1
H = 45 #10^-6 T

fig = plt.figure(figsize=(9.2, 7))
ax = a3.Axes3D(fig)
ax.grid(b=None)
ax.dist=10
ax.azim=30
ax.elev=10
C = T - dV/dS*P + dM/dS*H
print("CONSTANT",C)
T2=1300
P2=100
H2=(C-T2+dV/dS*P2)*dS/dM
T3=1400
P3=1000
H3=(C-T3+dV/dS*P3)*dS/dM
print(H2,H3)
limits = [[min(T,T2,T3)-100, max(T,T2,T3)+100],
          [min(P,P2,P3)-1, max(P,P2,P3)+1],
          [min(H,H2,H3)-1, max(H,H2,H3)+10]]


import numpy as np
from itertools import combinations

def add_plane_projection(ax,limits1):
    lines1 =[
        [[limits1[0][0],limits1[0][1]],[limits1[1][1],limits1[1][1]],[limits1[2][0]-blueinterval,limits1[2][0]-blueinterval]],
        [[limits1[0][0],limits1[0][1]],[limits1[1][0],limits1[1][0]],[limits1[2][0]-blueinterval,limits1[2][0]-blueinterval]],
        [[limits1[0][0],limits1[0][0]],[limits1[1][0],limits1[1][1]],[limits1[2][0]-blueinterval,limits1[2][0]-blueinterval]],
        [[limits1[0][1],limits1[0][1]],[limits1[1][0],limits1[1][1]],[limits1[2][0]-blueinterval,limits1[2][0]-blueinterval]],
        
        [[limits1[0][0]-greeninterval,limits1[0][0]-greeninterval],[limits1[1][0],limits1[1][0]],[limits1[2][0],limits1[2][1]]],
        [[limits1[0][0]-greeninterval,limits1[0][0]-greeninterval],[limits1[1][1],limits1[1][1]],[limits1[2][0],limits1[2][1]]],
        [[limits1[0][0]-greeninterval,limits1[0][0]-greeninterval],[limits1[1][0],limits1[1][1]],[limits1[2][0],limits1[2][0]]],
        [[limits1[0][0]-greeninterval,limits1[0][0]-greeninterval],[limits1[1][0],limits1[1][1]],[limits1[2][1],limits1[2][1]]],
        
        [[limits1[0][0],limits1[0][1]],[limits1[1][0]-orangeinterval,limits1[1][0]-orangeinterval],[limits1[2][0],limits1[2][0]]],
        [[limits1[0][0],limits1[0][1]],[limits1[1][0]-orangeinterval,limits1[1][0]-orangeinterval],[limits1[2][1],limits1[2][1]]],
        [[limits1[0][0],limits1[0][0]],[limits1[1][0]-orangeinterval,limits1[1][0]-orangeinterval],[limits1[2][0],limits1[2][1]]],
        [[limits1[0][1],limits1[0][1]],[limits1[1][0]-orangeinterval,limits1[1][0]-orangeinterval],[limits1[2][0],limits1[2][1]]],
        ]
    for i in lines1:
        ax.plot(i[0],i[1],i[2],color="black",linewidth=1)

blueinterval=100
greeninterval=1500
orangeinterval=1500
        
def add_axis_ticks(ax,limits1):
    ax.plot([limits1[0][0],limits1[0][1]],[limits1[1][1],limits1[1][1]],[limits1[2][0],limits1[2][0]],color="black",linewidth=0.7)
    interval=200
    per=0.02
    for i in range(limits1[0][0],limits1[0][1],interval):
        ax.plot([i,i],[limits1[1][1]*(1-per),limits1[1][1]*(1+per)],[limits1[2][0],limits1[2][0]],color="black",linewidth=0.7)
        ax.plot([i,i],[limits1[1][1]*(1-per)-orangeinterval-limits1[1][1],limits1[1][1]*(1+per)-orangeinterval-limits1[1][1]],[limits1[2][0],limits1[2][0]],color="black",linewidth=0.7)
        ax.plot([i,i],[limits1[1][1]*(1-per),limits1[1][1]*(1+per)],[limits1[2][0]-blueinterval,limits1[2][0]-blueinterval],color="black",linewidth=0.7)
    
    ax.plot([limits1[0][1],limits1[0][1]],[limits1[1][0],limits1[1][1]],[limits1[2][0],limits1[2][0]],color="black",linewidth=0.7)
    per=0.01
    for i in range(limits1[1][0],limits1[1][1],interval):
        ax.plot([limits1[0][1]*(1-per),limits1[0][1]*(1+per)],[i,i],[limits1[2][0],limits1[2][0]],color="black",linewidth=0.7)
        ax.plot([limits1[0][1]*(1-per),limits1[0][1]*(1+per)],[i,i],[limits1[2][0]-blueinterval,limits1[2][0]-blueinterval],color="black",linewidth=0.7)
        ax.plot([limits1[0][1]*(1-per)-greeninterval-1200,limits1[0][1]*(1+per)-greeninterval-1200],[i,i],[limits1[2][0],limits1[2][0]],color="black",linewidth=0.7)
    
    ax.plot([limits1[0][1],limits1[0][1]],[limits1[1][0],limits1[1][0]],[limits1[2][0],limits1[2][1]],color="black",linewidth=0.7)
    per=0.01
    for i in range(0,120,20):
        ax.plot([2000*(1-per),2000*(1+per)],[0,0],[i,i],color="black",linewidth=0.7)
        ax.plot([limits1[0][1]*(1-per)-greeninterval-1200,limits1[0][1]*(1+per)-greeninterval-1200],[0,0],[i,i],color="black",linewidth=0.7)
        ax.plot([2000*(1-per),2000*(1+per)],[limits1[1][1]*(1-per)-orangeinterval-limits1[1][1],limits1[1][1]*(1+per)-orangeinterval-limits1[1][1]],[i,i],color="black",linewidth=0.7)

limits1 = [[800,2000],[0,1000],[0,120]]
add_axis_ticks(ax,limits1)
add_plane_projection(ax,limits1)

'''add blue flat plane, gray flat plane, two blue lines'''
xx,yy = np.meshgrid(limits1[0],limits1[1])
ax.plot_surface(xx, yy, np.full_like(xx, limits1[2][0], dtype=np.double), 
                alpha=0.2,color=(0.95, 0.95, 0.95, 0.5))

# ax.plot_surface(xx, yy, np.full_like(xx, limits1[2][0]-blueinterval, dtype=np.double), 
#                 alpha=0.3,color='blue')
tris = [
    [[T,P,limits1[2][0]-blueinterval],
     [T+(P3-P)*dV/dS,P3,limits1[2][0]-blueinterval],
     [800,0,limits1[2][0]-blueinterval]],
    [[800,1000,limits1[2][0]-blueinterval],
     [T+(P3-P)*dV/dS,P3,limits1[2][0]-blueinterval],
     [800,0,limits1[2][0]-blueinterval]]
    ]
tris1 = [
    [[T,P,limits1[2][0]-blueinterval],
     [T+(P3-P)*dV/dS,P3,limits1[2][0]-blueinterval],
     [2000,0,limits1[2][0]-blueinterval]],
    [[2000,1000,limits1[2][0]-blueinterval],
     [T+(P3-P)*dV/dS,P3,limits1[2][0]-blueinterval],
     [2000,0,limits1[2][0]-blueinterval]]
    ]
for triss,fc in zip([tris,tris1],["blue","red"]):
    pc = a3.art3d.Poly3DCollection(triss, \
         alpha = 0.1, facecolor=fc,edgecolor = None)
    ax.add_collection3d(pc)
    
# blueinterval = -45
# tris = [
#     [[T,P,limits1[2][0]-blueinterval],
#      [T+(P3-P)*dV/dS,P3,limits1[2][0]-blueinterval],
#      [800,0,limits1[2][0]-blueinterval]],
#     [[800,1000,limits1[2][0]-blueinterval],
#      [T+(P3-P)*dV/dS,P3,limits1[2][0]-blueinterval],
#      [800,0,limits1[2][0]-blueinterval]]
#     ]
# tris1 = [
#     [[T,P,limits1[2][0]-blueinterval],
#      [T+(P3-P)*dV/dS,P3,limits1[2][0]-blueinterval],
#      [2000,0,limits1[2][0]-blueinterval]],
#     [[2000,1000,limits1[2][0]-blueinterval],
#      [T+(P3-P)*dV/dS,P3,limits1[2][0]-blueinterval],
#      [2000,0,limits1[2][0]-blueinterval]]
#     ]
# blueinterval = 100
# for triss,fc in zip([tris,tris1],["blue","red"]):
#     pc = a3.art3d.Poly3DCollection(triss, \
#          alpha = 0.1, facecolor=fc,edgecolor = None)
#     ax.add_collection3d(pc)
    
ax.plot([T,T+(650-P)*dV/dS],
        [P,650],
        [limits1[2][0]-blueinterval,limits1[2][0]-blueinterval],color="blue",linewidth=2)

a = Arrow3D([T,T+(680-P)*dV/dS],
        [P+20,680],
        [H,H], mutation_scale=10, zorder = 100,
                lw=2, arrowstyle="-|>", color="blue")
ax.add_artist(a)

'''add green vertical plane, green vertical plane, two green lines'''
yy,zz = np.meshgrid(limits1[1],limits1[2])
ax.plot_surface(np.full_like(yy, limits1[0][0], dtype=np.double), yy, zz, 
                alpha=0.1,color=(0.95, 0.95, 0.95, 0.5))

# ax.plot_surface(np.full_like(yy, limits1[0][0]-greeninterval, dtype=np.double), yy, zz, 
#                 alpha=0.3,color='green')

tris = [
    [[limits1[0][0]-greeninterval,P,H],
     [limits1[0][0]-greeninterval,P3,(P3-P)*dV/dM+H],
     [limits1[0][0]-greeninterval,0,120]],
    [[limits1[0][0]-greeninterval,1000,120],
     [limits1[0][0]-greeninterval,P3,(P3-P)*dV/dM+H],
     [limits1[0][0]-greeninterval,0,120]]
    ]
tris1 = [
    [[limits1[0][0]-greeninterval,P,H],
     [limits1[0][0]-greeninterval,P3,(P3-P)*dV/dM+H],
     [limits1[0][0]-greeninterval,0,0]],
    [[limits1[0][0]-greeninterval,1000,0],
     [limits1[0][0]-greeninterval,P3,(P3-P)*dV/dM+H],
     [limits1[0][0]-greeninterval,0,0]]
    ]
for triss,fc in zip([tris,tris1],["blue","red"]):
    pc = a3.art3d.Poly3DCollection(triss, \
         alpha = 0.1, facecolor=fc,edgecolor = None)
    ax.add_collection3d(pc)

# greeninterval = -383
# tris = [
#     [[limits1[0][0]-greeninterval,P,H],
#      [limits1[0][0]-greeninterval,P3,(P3-P)*dV/dM+H],
#      [limits1[0][0]-greeninterval,0,120]],
#     [[limits1[0][0]-greeninterval,1000,120],
#      [limits1[0][0]-greeninterval,P3,(P3-P)*dV/dM+H],
#      [limits1[0][0]-greeninterval,0,120]]
#     ]
# tris1 = [
#     [[limits1[0][0]-greeninterval,P,H],
#      [limits1[0][0]-greeninterval,P3,(P3-P)*dV/dM+H],
#      [limits1[0][0]-greeninterval,0,0]],
#     [[limits1[0][0]-greeninterval,1000,0],
#      [limits1[0][0]-greeninterval,P3,(P3-P)*dV/dM+H],
#      [limits1[0][0]-greeninterval,0,0]]
#     ]
# greeninterval = 1500
# for triss,fc in zip([tris,tris1],["red","blue"]):
#     pc = a3.art3d.Poly3DCollection(triss, \
#          alpha = 0.1, facecolor=fc,edgecolor = None)
#     ax.add_collection3d(pc)
ax.plot([limits1[0][0]-greeninterval,limits1[0][0]-greeninterval],
        [P,650],
        [H,(650-P)*dV/dM+H],color="green",linewidth=2)
a = Arrow3D([T,T],
        [P+10,680],
        [H,(680-P)*dV/dM+H],color="green",mutation_scale=10, zorder = 100,
                lw=2, arrowstyle="-|>")
ax.add_artist(a)

'''add orange vertical plane, orange vertical plane, two orange lines'''
xx,zz = np.meshgrid(limits1[0],limits1[2])
ax.plot_surface(xx, np.full_like(xx, limits1[1][0], dtype=np.double), zz, 
                alpha=0.2,color=(0.95, 0.95, 0.95, 0.5))
# ax.plot_surface(xx, np.full_like(xx, limits1[1][0]-orangeinterval, dtype=np.double), zz, 
#                 alpha=0.3,color='orange')
tris = [
    [[928.23,limits1[1][0]-orangeinterval,0],
     [T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
     [800,limits1[1][0]-orangeinterval,0]],
    [[T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
     [800,limits1[1][0]-orangeinterval,120],
     [800,limits1[1][0]-orangeinterval,0]]
    ]
tris1 = [
    [[928.23,limits1[1][0]-orangeinterval,0],
     [T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
     [2000,limits1[1][0]-orangeinterval,0]],
    [[2000,limits1[1][0]-orangeinterval,120],
     [T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
     [2000,limits1[1][0]-orangeinterval,0]],
    ]
for triss,fc in zip([tris,tris1],["blue","red"]):
    pc = a3.art3d.Poly3DCollection(triss, \
         alpha = 0.1, facecolor=fc,edgecolor = None)
    ax.add_collection3d(pc)

# orangeinterval = 0
# tris = [
#     [[928.23,limits1[1][0]-orangeinterval,0],
#      [T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
#      [800,limits1[1][0]-orangeinterval,0]],
#     [[T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
#      [800,limits1[1][0]-orangeinterval,120],
#      [800,limits1[1][0]-orangeinterval,0]]
#     ]
# tris1 = [
# #     [[928.23,limits1[1][0]-orangeinterval,0],
# #      [T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
# #      [2000,limits1[1][0]-orangeinterval,0]],
#     [[2000,limits1[1][0]-orangeinterval,120],
#      [T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
#      [2000,limits1[1][0]-orangeinterval,0]],
#     ]
# 
# orangeinterval = 1500
# for triss,fc in zip([tris,tris1],["blue","red"]):
#     pc = a3.art3d.Poly3DCollection(triss, \
#          alpha = 0.1, facecolor=fc,edgecolor = None)
#     ax.add_collection3d(pc)
    
ax.plot([T,T-(H3-H)*dM/dS],
        [limits1[1][0]-orangeinterval,limits1[1][0]-orangeinterval],
        [H,H3],color = "orange",linewidth=2.5)
a = Arrow3D([T,T-(H3-H)*dM/dS+20],
        [min(P,P2,P3)-1,min(P,P2,P3)-1],
        [H,H3+3],color = "orange",mutation_scale=10, zorder = 100,
                lw=2, arrowstyle="-|>")
ax.add_artist(a)

'''add dash lines from line in 3d to 2d'''
lines = [
    [[T,T],[P,P],[H,limits1[2][0]-blueinterval]],
    [[T+(650-P)*dV/dS,T+(650-P)*dV/dS],[650,650],[H,limits1[2][0]-blueinterval]],
    [[T,limits1[0][0]-greeninterval],[P,P],[H,H]],
    [[T,limits1[0][0]-greeninterval],[650,650],[(650-P)*dV/dM+H,(650-P)*dV/dM+H]],
    [[T,T],[min(P,P2,650)-1,limits1[1][0]-orangeinterval],[H,H]],
    [[T-(H3-H)*dM/dS,T-(H3-H)*dM/dS],[min(P,P2,650)-1,limits1[1][0]-orangeinterval],[H3,H3]]
]
for line in lines:
    ax.plot(line[0],line[1],line[2],linestyle="--",color="gray",linewidth=1,
            alpha=0.6)
    
# limits1 = [[800,2000],[0,1000],[0,120]]
tt = [1000,1600]
pp = [-200,1000]
# print(pp)
xx,yy = np.meshgrid(tt,pp)
normal = [1, -dV/dS, dM/dS,-C]

# add 3d polytope for red phase
orangeinterval = 0
vertices = [[928.23,limits1[1][0]-orangeinterval,0],
     [T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
     [2000,limits1[1][0]-orangeinterval,0],
     [2000,limits1[1][0]-orangeinterval,120],
     [2000,limits1[1][1],0],
     [2000,limits1[1][1],120],
     [-normal[1]*1000-normal[2]*120-normal[3],1000,120],
     [-normal[1]*1000-normal[2]*0-normal[3],1000,0],
     ]
vertices = np.array(vertices)
from scipy.spatial.qhull import ConvexHull
hull = ConvexHull(vertices)
simplices = hull.simplices 
for s in simplices:
    print(s)
    print(vertices[s])
org_triangles = [vertices[s] for s in simplices]

pc = a3.art3d.Poly3DCollection(org_triangles, \
     alpha = 0.03, facecolor="red",edgecolor = None)
ax.add_collection3d(pc)
# add red phase end
# add 3d polytope for blue phase
vertices = [
         [928.23,limits1[1][0]-orangeinterval,0],
         [T-(120-H)*dM/dS,limits1[1][0]-orangeinterval,120],
         [800,limits1[1][0]-orangeinterval,120],
         [800,limits1[1][0]-orangeinterval,0],
         [800,limits1[1][1],0],
         [800,limits1[1][1],120],
         [-normal[1]*1000-normal[2]*120-normal[3],1000,120],
         [-normal[1]*1000-normal[2]*0-normal[3],1000,0],
     ]
vertices = np.array(vertices)
from scipy.spatial.qhull import ConvexHull
hull = ConvexHull(vertices)
simplices = hull.simplices 
for s in simplices:
    print(s)
    print(vertices[s])
org_triangles = [vertices[s] for s in simplices]

pc = a3.art3d.Poly3DCollection(org_triangles, \
     alpha = 0.03, facecolor="blue",edgecolor = None)
ax.add_collection3d(pc)
# add blue phase end

z = (-normal[0]*xx-normal[1]*yy-normal[-1])* 1. /normal[2]
print(z)
ax.plot_surface(xx, yy, z, alpha=0.8,color="#8afbff",zorder=50)
ax.scatter(T,P,H,c="black",zorder=200,s=50)
newlimits = [[0,2000],[-600,1000],[-20,140]]
expandfactor = 1.5
newlimits = [[np.average(i)-1/2*expandfactor*(max(i)-min(i)),np.average(i)+1/2*expandfactor*(max(i)-min(i))] for i in newlimits]
print(newlimits)
# for i in newlimits:
#     [np.average(i)-1/2*(max(i)-min(i)),np.average(i)+1/2*(max(i)-min(i))]
ax.set_xlim(newlimits[0])
ax.set_ylim(newlimits[1])
ax.set_zlim(newlimits[2])
ax.set_xlabel("T (K)")
ax.set_ylabel("P (bar)")
ax.set_zlabel("H ($\mu$T)")


ax.set_axis_off()
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.show()




















