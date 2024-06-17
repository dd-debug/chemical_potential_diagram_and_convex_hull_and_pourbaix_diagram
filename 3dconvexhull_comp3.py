'''
Created on Jun 5, 2021

@author: jiadongc
'''



from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.periodic_table import Element

'''import new_PDPlotter'''
from phase_diagram_packages.convexhullpdplotter import new_PDPlotter,getOrigStableEntriesList

# ternary compound convex hull such as Figure 3a, 4a in Duality paper.
# if you want to change color of materials, add a new colordict in get_marker_props method from new_PDPlotter
elsList = ["Co","Ni","O"]
elementlist = [Element(el) for el in elsList]
PDentries = getOrigStableEntriesList(elsList)

pd = PhaseDiagram(PDentries,elementlist)

new_PDPlotter(pd, ternary_style = "3d").show()
# new_PDPlotter(pd).add_line()

# {'Co3Ni': [0.75, 0.0, 0.25, 1.0], 'CrNi2': [-0.0, 0.3333, 0.6667, 1.0], 'Ni': [0.0, 0.0, 1.0, 1.0], 'Cr': [0.0, 1.0, 0.0, 1.0], 'Co': [1.0, 0.0, 0.0, 1.0], 'Cr2O3': [0.0, 1.0, 0.0, 0.4233], 'Cr2CoO4': [0.3333, 0.6667, 0.0, 0.3447], 'CoO': [1.0, 0.0, 0.0, 0.2923], 'NiO': [0.0, 0.0, 1.0, 0.2797], 'Co3O4': [1.0, 0.0, 0.0, 0.1603], 'CrNiO4': [0.0, 0.5, 0.5, 0.1029], 'Co(NiO2)2': [0.3333, 0.0, 0.6667, 0.0928], 'CrO2': [0.0, 1.0, 0.0, 0.0908], 'CoNiO3': [0.5, 0.0, 0.5, 0.0907], 'CoO2': [1.0, 0.0, 0.0, 0.0874], 'CrCoO4': [0.5, 0.5, 0.0, 0.0781], 'Cr5O12': [0.0, 1.0, 0.0, 0.0682], 'Ni3O4': [0.0, 0.0, 1.0, 0.0494]}
