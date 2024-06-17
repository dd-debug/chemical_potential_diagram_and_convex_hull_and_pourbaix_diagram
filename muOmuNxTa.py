# -*- coding: UTF-8 -*-
'''
Created on Mar 21, 2023

@author: jiadongc
'''
from phase_diagram_packages.ChemicalPotentialDiagram import ChemPotDiagram,ChemPotEntry, \
ChemPotPlotter,trans_PD_to_ChemPot_entries
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from phase_diagram_packages.convexhullpdplotter import getOrigStableEntriesList

# Figure 3b in the duality paper

def make_entry_from_formEperatom(pd, Composition, formEperatom):
    from pymatgen.entries.computed_entries import ComputedEntry
    c = Composition
    EntryE=formEperatom*c.num_atoms+ sum([c[el]*pd.el_refs[el].energy_per_atom
                                   for el in c.elements])
    new_entry=ComputedEntry(Composition, EntryE)
    return new_entry

compound = "TaON"
elsList = [str(i) for i in Composition(compound).elements]

projEle = ['O','N']

PDentries = getOrigStableEntriesList(elsList)

newformeN = 1
newformeO = 1
CPentries = trans_PD_to_ChemPot_entries(PDentries,elsList)
for e in PhaseDiagram(PDentries).stable_entries:
    if e.name == "N2":
        entry = e
        entry = make_entry_from_formEperatom(PhaseDiagram(PDentries), e.composition, newformeN)
    if e.name == "O2":
        entryO = e # seems like no difference
        entryO = make_entry_from_formEperatom(PhaseDiagram(PDentries), e.composition, newformeO)
CPentries = trans_PD_to_ChemPot_entries(PDentries,elsList)
limits = [[-8, 0],[-8,newformeN],[-8,newformeO]]
limits = [[-10,0],[-6,0],[-6,0]]
newentries = []
for e in CPentries:
    if e.name not in ["N2O", "NO3", "NO2", "N2O5"]:
        newentries.append(e)
newentries.append(ChemPotEntry(entry,newformeN,elsList))
newentries.append(ChemPotEntry(entryO,newformeO,elsList))
cp = ChemPotDiagram(newentries,elsList,limits = limits)


plotter = ChemPotPlotter(cp)
plotter.get_projection_equilLine_mu_x_plot(projEle, \
    limits = limits,show_polytope = False,
    alpha = 0.3,label_equilLine = 1)
