'''
Created on Jun 5, 2021

@author: jiadongc
'''

from phase_diagram_packages.ChemicalPotentialDiagram import ChemPotDiagram,ChemPotPlotter,trans_PD_to_ChemPot_entries
from phase_diagram_packages.convexhullpdplotter import getOrigStableEntriesList

'''codes to produce chemical potential diagram
Figure 2d, use this code you can visualize 3-component system,
it can also extend to higher dimensional
'''

elsList = ["Mn","O"] 
PDentries = getOrigStableEntriesList(elsList)
# pd = PhaseDiagram(PDentries)
# new_PDPlotter(pd).show()

limits = [[-7,0],[-6,0]]

CPentries = trans_PD_to_ChemPot_entries(PDentries,elsList)

cp = ChemPotDiagram(CPentries,elsList,limits = limits)

ChemPotPlotter(cp).get_equil_line_on_CP_Halfspace(limits = limits,
    alpha = 0.2,show_polytope = 1, show_label=1,
    label_domain = 1, label_equilLine = 0)
