
from pylab import contour,colorbar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap

#http://glowingpython.blogspot.com/2012/01/how-to-plot-two-variable-functions-with.html

# the function that I'm going to plot
def approx_z_func(T,pO2):
    kB=8.617*10**-5 #eV/K
    pO2=10**pO2
    deltaE=-5/2*kB*T*np.log(T)+kB*T*np.log(pO2)
    #deltaE=-.2*96.48*T+kB*T*np.log(pO2)
    return 0.442345+deltaE

def karenjoh_z_func(T,pO2):
    A=29.659*10**-3
    B=6.137261*10**-6
    C=-1.186521*10**-9
    D=0.095780*10**-12
    E=-0.219663*10**3
    F=-9.861391
    G=237.948*10**-3
    
    G_T=A*(T-T*np.log(T))-B*T**2/2.0-C*T**3/6.0-D*T**4/12.0-E/2.0/T+F-G*T
    T0=300
    G0_T0=A*(T0-T0*np.log(T0))-0.5*B*T0**2-C*T0**3/6.0-D*T0**4/12.0-E/2.0/T0+F-G*T0
    dG=(G_T-G0_T0)/96.48
    
    kB=8.617*10**-5 #eV/K
    deltaE=-5/2*kB*T*np.log(T/300)+kB*T*np.log(pO2)
    pO2=10**pO2
    #print pO2
    deltaE=dG+kB*T*np.log(pO2)
    return deltaE

#http://www.physics.rutgers.edu/~karenjoh/manuscript.pdf
#http://webbook.nist.gov/cgi/inchi?ID=C7782447&Mask=1#Thermo-Gas

def z_func(T,pO2):
    
    #if T<700:
    A=31.32234
    B=-20.23531
    C=57.86644
    D=-36.50624
    E=-0.007374
    F=-8.903471
    G=246.7945
    H=0.0

    A=30.03235
    B=8.772972
    C=-3.988133
    D=0.788313
    E=-0.741599
    F=-11.32468
    G=236.1663
    H=0.0

    t=T/1000.0
    dH = A*t + B*t**2/2.0 + C*t**3/3.0 + D*t**4/4.0 - E/t + F
    S = A*np.log(t) + B*t + C*t**2/2.0 + D*t**3/3.0 - E/(2*t**2) + G
    dG=dH/1000/96.48-T*S/1000/96.48
    kB=8.617*10**-5 #eV/K
    #deltaE=-5/2*kB*T*np.log(T/300)+kB*T*np.log(pO2)
    pO2=10**pO2
    deltaE=dG+kB*T*np.log(pO2)
    #return -4.935+0.64+deltaE
    return 0.64+deltaE
def plot_meshgrid_region(cset, 
                         xlimit, ylimit, zlimit,
                         color,alpha = 0.5):
    # cset is the corresponding contour of zlimit
    zlimit.sort()
    xlimit.sort()
    ylimit.sort()
            
    # smaller z
    x = cset.allsegs[0][0][:,0] if len(cset.allsegs[0]) else []
    y = cset.allsegs[0][0][:,1] if len(cset.allsegs[0]) else []
    
    if len(x) and min(x) < xlimit[0]:
        index = x >= xlimit[0]
        x = x[index]
        y = y[index]
    if len(x) and max(x) > xlimit[1]:
        index = x <= xlimit[1]
        x = x[index]
        y = y[index]
    if len(y) and min(y) < ylimit[0]:
        index = y >= ylimit[0]
        y = y[index]
        x = x[index]
    if len(y) and max(y) > ylimit[1]:
        index = y <= ylimit[1]
        y = y[index]
        x = x[index]
        
    # larger z
    x1 = cset.allsegs[1][0][:,0] if len(cset.allsegs[1]) else [] 
    y1 = cset.allsegs[1][0][:,1] if len(cset.allsegs[1]) else []
    if len(x1) and min(x1) < xlimit[0]:
        index = x1 >= xlimit[0]
        x1 = x1[index]
        y1 = y1[index]
    if len(x1) and max(x1) > xlimit[1]:
        index = x1 <= xlimit[1]
        x1 = x1[index]
        y1 = y1[index]
    if len(y1) and min(y1) < ylimit[0]:
        index = y1 >= ylimit[0]
        y1 = y1[index]
        x1 = x1[index]
    if len(y1) and max(y1) > ylimit[1]:
        index = y1 <= ylimit[1]
        y1 = y1[index]
        x1 = x1[index]
        
    # å°†åœ¨zèŒƒå›´å†…çš„x yåŠ å…¥ç‚¹
    ends = [[xlimit[0],ylimit[0]],[xlimit[0],ylimit[1]],
     [xlimit[1],ylimit[0]],[xlimit[1],ylimit[1]]]
    for t,p in ends:
        z = z_func(t,p)
        print(t,p,z,zlimit[0],zlimit[1])
        if zlimit[0] <= z <= zlimit[1]:
            print(len(x1))
            x1 = np.append(x1, t)
            y1 = np.append(y1, p)
            print(len(x1))

    # è®¡ç®—æ¯�ä¸ªæ•°ç»„çš„ä¸­å¿ƒç‚¹å��æ ‡

    xx = np.concatenate((x, x1), axis=0)
    yy = np.concatenate((y, y1), axis=0)

    arr = np.column_stack((xx,yy))
    refer = np.array([xx[-1],yy[-1]])

    # è®¡ç®—æ¯�ä¸ªç‚¹ç›¸å¯¹äºŽä¸­å¿ƒç‚¹çš„æž�è§’ï¼ˆè§’åº¦ï¼‰
    angles = np.arctan2(arr[:, 1] - refer[1], arr[:, 0] - refer[0])
    sorted_indices = np.argsort(angles)
    sorted_points = arr[sorted_indices]

    plt.fill(sorted_points[:, 0], sorted_points[:, 1],
             color,alpha = alpha)

      
def get_color_map():
    bottom = cm.get_cmap('Blues', 128)
    
    top = cm.get_cmap('Reds_r', 128)
#     print(np.linspace(0, 1, 128))
#     print(top(np.linspace(0, 1, 128)))
    newcolors = np.vstack((top(np.linspace(0, 1, 128))[int(128/10*(10-2)):],
                           bottom(np.linspace(0, 1, 128))))
    newcolors = newcolors[::-1]
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    return newcmp
 

print(z_func(298,0))

pO2 = np.arange(-12, 10, 0.01)
T = np.arange(100, 2500, 100)
X, Y = np.meshgrid(T, pO2)
print(X.shape,Y.shape)
Z = z_func(X, Y) # evaluation of the function on the grid


zz = []
for t,p in zip([300,600,900,1200,1500],[5,5,5,5,5]):
    zz.append(z_func(t,p))
print(zz)
  
  
zz = []
for t,p in zip([300,600,900,1200,1500],[0,0,0,0,0]):
    zz.append(z_func(t,p))
print(zz)
  
zz = []
for t,p in zip([300,600,900,1200,1500],[-5,-5,-5,-5,-5]):
    zz.append(z_func(t,p))
print(zz)
  
zz = []
for t,p in zip([300,600,900,1200,1500],[-8]*5):
    zz.append(z_func(t,p))
print(zz)
  
#print np.shape(Z)
fig = plt.figure(figsize=(11, 7))


orig_cmap = matplotlib.cm.PuBu_r
#shifted_cmap = shiftedColorMap(orig_cmap,start=-5, midpoint=0, stop=1)

im = plt.contourf(X,Y,Z,300,cmap=orig_cmap) # drawing the function
# adding the Contour lines with labels
cset = contour(X,Y,Z,np.arange(-10,1,1),linewidths=1,colors='k')
cset1 = contour(X,Y,Z,np.arange(0.5,1,0.5),linewidths=1,colors='k')


plt.xlabel('Temperature (K)')
plt.ylabel('log10(pO2)')

plt.clabel(cset,inline=1,fmt='%1.1f',fontsize=15, manual = [(180,-7),(440,-7),(690,-7),
                                                        (930,-7),(1200,-7),(1400,-7),
                                                           (1600,-7),(1840,-7),(2100,-7),
                                                           (2300,-7)])
plt.clabel(cset1,inline=1,fmt='%1.1f',fontsize=15)
colorbar(im).set_ticks(np.arange(-10,0.1,2)) # adding the colobar on the right
plt.scatter([298.15], [0], c="k",marker = "*",s=300)
# plt.plot([min(cset2.allsegs[1][0][:,0]),min(cset2.allsegs[1][0][:,0])],
#          [-8,0.99],linestyle = "dashed",c = "gray",alpha = 1,linewidth = 1)
# plt.plot([max(cset3.allsegs[0][0][:,0]),max(cset3.allsegs[0][0][:,0])],
#          [-8,0.99],linestyle = "dashed",c = "gray",alpha = 1,linewidth = 1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(100, 1000)
plt.ylim(-10, 10)

# latex fashion title
plt.show()