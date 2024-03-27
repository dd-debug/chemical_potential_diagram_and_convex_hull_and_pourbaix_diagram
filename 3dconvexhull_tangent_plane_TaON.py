'''
Created on Jun 5, 2021

@author: jiadongc
'''

from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.periodic_table import Element
from phase_diagram_packages.convexhullpdplotter import new_PDPlotter,getOrigStableEntriesList


elsList = ["O","N","Ta"]
elementlist = [Element(el) for el in elsList]
PDentries = getOrigStableEntriesList(elsList)
newentries = []
for e in PDentries:
    if e.name not in ["N2O", "NO3", "NO2", "N2O5"]:
        newentries.append(e)
pd = PhaseDiagram(newentries,elementlist)
# PDPlotter(pd).show()
new_PDPlotter(pd,ternary_style = "3d").show(
                       add_tangent_plane = "TaNO",show_elements = False,
                       irpd = [
                             [Composition("Ta2O5"),Composition("Ta3N5")],
                             ])
# new_PDPlotter(pd).add_line()

