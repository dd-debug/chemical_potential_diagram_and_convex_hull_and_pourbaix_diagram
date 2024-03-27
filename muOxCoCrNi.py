'''
Created on Mar 21, 2023

@author: jiadongc
'''
from phase_diagram_packages.convexhullpdplotter import getOrigStableEntriesList

from phase_diagram_packages.ChemicalPotentialDiagram import ChemPotDiagram,ChemPotPlotter,trans_PD_to_ChemPot_entries


els = ["Co", "Cr", "Ni", "O"]
entries = getOrigStableEntriesList(els)
limits = [[-10,0],[-10,0],[-10,0],[-10,0]]
CPentries = trans_PD_to_ChemPot_entries(entries,els)

cp = ChemPotDiagram(CPentries,els,limits = limits)
plotter = ChemPotPlotter(cp)
projEle = ["O"]
plotter.get_mu_xxx_plot2(projEle, 
    show_range = [-2, 0],
    limits = limits,
    alpha = 0.2,
    label_single = 1,
    axis_off = 1)

# chempots = {"N":-1}
# gpd = GraPotPhaseDiagram(entries, chempots)
# PDPlotter(gpd).show()
