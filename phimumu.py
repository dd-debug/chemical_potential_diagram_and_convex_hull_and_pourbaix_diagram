# -*- coding: UTF-8 -*-
'''
Created on Mar 21, 2023

@author: jiadongc
'''

from phase_diagram_packages.ChemicalPotentialDiagram import ChemPotDiagram,ChemPotPlotter,trans_PD_to_ChemPot_entries
from phase_diagram_packages.convexhullpdplotter import getOrigStableEntriesList


# elsList = ['Ba','Fe','Mn'] #no stable materials except single matters
elsList = ["Mn",'O'] 
# PDentries = MPR.get_entries_in_chemsys(elsList)
PDentries = getOrigStableEntriesList(elsList)
limits = [[-7,0], [-6,0]]
CPentries = trans_PD_to_ChemPot_entries(PDentries,elsList)
cp = ChemPotDiagram(CPentries,elsList,limits = limits)
plotter = ChemPotPlotter(cp)
limits = [[-7,0],[-6,0],[-14,8]]
# plotter.show()
plotter.get_phimumu_plot(alpha = 0.3,limits = limits,
                         show_label=1,label_equiline=1)
# [-4.05734142 -3.94265858  1.92447197]
