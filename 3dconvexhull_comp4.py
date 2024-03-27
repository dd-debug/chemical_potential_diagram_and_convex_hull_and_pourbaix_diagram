'''
Created on Jun 5, 2021
 
@author: jiadongc
'''

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.periodic_table import Element

'''import new_PDPlotter'''
from phase_diagram_packages.convexhullpdplotter import new_PDPlotter,getOrigStableEntriesList
 
 
elsList = ["Co","Ni","Cr","O"]
# elsList = ["Co", "Li", "Ni", "O"]
elementlist = [Element(el) for el in elsList]
PDentries = getOrigStableEntriesList(elsList)
 
pd = PhaseDiagram(PDentries,elementlist)
 
new_PDPlotter(pd, ternary_style = "3d").show(
    show_elements = 0,
    add_3in4 = [["NiO", "CoO","Cr2O3"],
                ["CoO2", "CrNiO4", "CrO2"]],
    add_triangle_colormap = 1
    )