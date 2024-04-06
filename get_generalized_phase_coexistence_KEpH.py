'''
Created on 2022.02.10
 
@author: jdche
'''
from phase_diagram_packages.GeneralizedChemPotDiagram import GeneralizedEntry,GeneralizedDiagram,GeneralizedPlotter
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition
from itertools import combinations
# from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram
 
els = ["Mn","O","H","K"]
Ec = -0.0594573 *(1.6) #(eV/Mn atom)
# Ec = 0
def build_generalized_entries_from_txt(
        els=els,filename="energies_info_processed.txt"):
    names = []
    formEs = []
     
    with open("entries_data/"+filename,"r") as f:
        energies =f.readlines()
    surf_product_d = {}
    for i in energies:
        if "#" in i or i == "\n":
            continue
        list1 = i.split(" ")
        names.append(list1[0])
        if "{" in list1[0]:
            formEs.append(float(list1[1]) + Ec)
            # print(list1[0])
        else:
            formEs.append(float(list1[1]))
        if len(list1) > 2:
            surf_product_d[list1[0]] = float(list1[-1])
         
    for key in surf_product_d: #1/R nm
        surf_product_d[key] = surf_product_d[key]*6.24*10**(-3)
    # print(surf_product_d)
     
    entries = []
    for i,j in zip(names,formEs):
        surf_p=0
        if i in surf_product_d:
            surf_p = surf_product_d[i]
         
        entry = GeneralizedEntry(i,j,els,surf_product=surf_p)
        entries.append(entry)
     
    return entries
 
entries = build_generalized_entries_from_txt()
fixed = [("Mn",0)]
limits = [[-10,0],[-6,2],[-10,0],[-10,0]]
limits += [[0,2],[0,1]]
# vislimits = [[-10,0],[-6,2],[-10,0],[-9,-5]]
# vislimits += [[0,1],[0,0.6]]
gpd = GeneralizedDiagram(entries,els,fixed=fixed, limits=limits,
                         surf=True,normalizedfactor="Nm",
                         mutoPH=True,
                         slicePlane=True,sliceLower=False,
                         w_aqu_constraint=True)
# dict1= gpd.get_phase_coexistence_region_w_aqueous_constraint()
# for i in dict1["phase5"]:
#     print(i[0])
 

gpdplotter = GeneralizedPlotter(gpd,fixed=fixed)
axiss = ["1/R","PH","K","E"]
for vis_axis in list(combinations(axiss,2)):
    print("axiss",vis_axis)

vislimits = [[-10,0],[-6,2],[-10,0],[-8,-2]]
vislimits += [[0,1],[0,0.6]]
# list1 = ["green","blue","red","orange","black","purple"]
# from itertools import permutations
# for fcs in list(permutations(list1, 4)):
 
gpdplotter.get_phase_coexistence(
    limits=limits,alpha=0.1,
    bold_boundary=True,
    label_domains=False,
    vis_axis=["PH","E","K"],
    vis_phase2 = 0,
    vis_phase3 = 1,
    vis_phase4 = 0,
    vis_phase5 = 0,
    vis_phase1 = ['delta-K0.21MnO1.87',"Mn^{2+}","beta-MnO2","alpha-K0.11MnO1.94"])
#     fcs = ["green","blue","red","orange"])
plt.show()
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  

