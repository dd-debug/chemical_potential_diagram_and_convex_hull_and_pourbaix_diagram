'''
Created on Jun 5, 2022

@author: jiadongc
'''



from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, CompoundPhaseDiagram

'''import new_PDPlotter'''
from phase_diagram_packages.convexhullpdplotter import new_PDPlotter,getOrigStableEntriesList

# Figure 5b in duality paper

terminals = [Composition("NiO"), Composition("CoO"), Composition("Cr2O3")]
# terminals = [Composition("CoO2"), Composition("CrNiO4"), Composition("CrO2")]
els = ["Co","Ni","Cr","O"]
entries = getOrigStableEntriesList(els)

pd = PhaseDiagram(entries)


cpd = CompoundPhaseDiagram(entries,terminals)


new_PDPlotter(cpd, ternary_style = "3d").show(marksize = 20,
                        show_elements = 1)

