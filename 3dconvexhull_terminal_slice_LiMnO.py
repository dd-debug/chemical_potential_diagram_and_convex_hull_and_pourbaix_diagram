'''
Created on Jun 5, 2021

@author: jiadongc
'''
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.periodic_table import Element
from phase_diagram_packages.convexhullpdplotter import new_PDPlotter,getOrigStableEntriesList

# Figure 4a, 4b in Duality paper

elsList = ["Li","Mn","O"]
elementlist = [Element(el) for el in elsList]
PDentries = getOrigStableEntriesList(elsList)
pd = PhaseDiagram(PDentries,elementlist)

# add_terminal_plane will add a colored rectangle
# irpd adds the reaction compound convex hull
new_PDPlotter(pd,ternary_style = "3d").show(
                        add_terminal_plane = 1,
                        show_elements = 1,
                        irpd = [
                             [Composition("O2"),Composition("LiMn2")],
                             ])
# new_PDPlotter(pd).add_line()

