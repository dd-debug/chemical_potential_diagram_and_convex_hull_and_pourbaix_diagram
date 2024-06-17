'''
Created on Dec 2, 2021

@author: jiadongc
'''
from phase_diagram_packages.GeneralizedChemPotDiagram import GeneralizedEntry,GeneralizedDiagram,GeneralizedPlotter
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition
# from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram

# Figure 5a in generalized Clausius paper

els = ["Mn","O","H"]
names = ["Mn","Mn^{2+}","Mn^{3+}","MnO4^{1-}",'MnO4^{2-}','Mn(OH)3^{1-}',
         'MnOH^{1+}','HMnO2^{1-}','Mn(OH)2',"\beta-MnOOH",'Mn3O4','\alpha-MnOOH',
         '\gamma-MnOOH',"Mn2O3",'R-MnO2','\gamma-MnO2','\beta-MnO2']
names = ["Mn","Mn^{2+}","Mn^{3+}","MnO4^{1-}",'MnO4^{2-}','Mn(OH)3^{1-}',
         'MnOH^{1+}','HMnO2^{1-}','Mn(OH)2',"beta-MnOOH",'Mn3O4','alpha-MnOOH',
         'gamma-MnOOH',"Mn2O3",'R-MnO2','gamma-MnO2','beta-MnO2']
formEs = [0,-2.363,-0.850,-4.658,-5.222,-7.714,-4.198,-5.243,-6.381,-5.629,-13.300,-5.763,-5.780,-9.132,-4.767,-4.787,-4.821]
entries = []
for i,j in zip(names,formEs):
    entry = GeneralizedEntry(i,j,els)
    entries.append(entry)
fixed = [("Mn",0)]
limits = [[-10,0],[-6,0],[-10,0]]
limits += [[-2,2]]
gpd = GeneralizedDiagram(entries,els,fixed=fixed, limits=limits,
                         normalizedfactor="Nm",mutoPH=True)
gpdplotter = GeneralizedPlotter(gpd,fixed=fixed)
gpdplotter.get_pourbaix_diagram(limits=limits,alpha=0.25,
                                show_phi=False,label_domains=1)

