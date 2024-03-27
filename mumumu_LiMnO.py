'''
Created on Jun 5, 2021

@author: jiadongc
'''

from phase_diagram_packages.ChemicalPotentialDiagram import ChemPotDiagram,ChemPotPlotter,trans_PD_to_ChemPot_entries

from phase_diagram_packages.convexhullpdplotter import getOrigStableEntriesList

elsList = ["Mn","Li","O"] 
PDentries = getOrigStableEntriesList(elsList)
# pd = PhaseDiagram(PDentries)
# new_PDPlotter(pd).show()

limits = [[-7,0],[-6,0],[-6,0]]

CPentries = trans_PD_to_ChemPot_entries(PDentries,elsList)

cp = ChemPotDiagram(CPentries,elsList,limits = limits)

ChemPotPlotter(cp).get_equil_line_on_CP_Halfspace(limits = limits,
    alpha = 0.2,show_polytope = 1, show_label=1,
    label_domain = 1, label_equilLine = 0)
