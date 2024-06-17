'''
Created on 09.03.2021 laptop

@author: jiadongc
'''
from phase_diagram_packages.ChemicalPotentialDiagram import trans_PD_to_ChemPot_entries, ChemPotDiagram, ChemPotPlotter
from phase_diagram_packages.convexhullpdplotter import getOrigStableEntriesList

# Figure 4d - 4f in duality paper

els = ["Mn","Li","O"]
el_mu="Li"
entries = getOrigStableEntriesList(els)

limits = [[-10,0],[-4.5,0],[-10,0]]
# limits = [[-20,0],[-10,0],[-10,0]]
CPentries = trans_PD_to_ChemPot_entries(entries,els)
cp = ChemPotDiagram(CPentries,els,limits = limits)
projEle = [el_mu]
plotter = ChemPotPlotter(cp)
plotter.get_mu_xx_plot_reverse(projEle, 
    limits = limits,show_polytope = False,
    alpha = 0.3,label_domains = 1)

