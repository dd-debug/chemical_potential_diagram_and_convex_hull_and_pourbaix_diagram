# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import plotly.graph_objects as go
import logging
from phase_diagram_packages.EquilibriumLine import EquilLine
from phase_diagram_packages.PhaseCoexistence import Phase23,Phase1,compare_xy_lines,compare_xy_vertices
import itertools

from functools import cmp_to_key, partial, lru_cache
from monty.json import MSONable, MontyDecoder


from scipy.spatial import ConvexHull, HalfspaceIntersection
import numpy as np

from itertools import combinations

from adjustText import adjust_text
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from phase_diagram_packages.GrandPotDiagram import GraPotPhaseDiagram

try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from pymatgen.util.coord import Simplex


from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition

from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter

import math
# MPR = MPRester("2d5wyVmhDCpPMAkq")
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

__author__ = "Jiadong Chen"
__copyright__ = "Copyright 2020"
__version__ = "0.1"
__maintainer__ = "Jiadong Chen"
__credits__ = "Jiadong Chen"
__email__ = "jiadongc@umich.edu"
__status__ = "Production"
__date__ = "June 12th, 2020"

logger = logging.getLogger(__name__)

MU_H2O = -2.4583
PREFAC = 0.0591


# TODO: Revise to more closely reflect PDEntry, invoke from energy/composition
# TODO: PourbaixEntries depend implicitly on having entry energies be
#       formation energies, should be a better way to get from raw energies
# TODO: uncorrected_energy is a bit of a misnomer, but not sure what to rename
def trans_PD_to_ChemPot_entries(PDentries,elslist):
    '''
    transform PDentries to ChemPotEntries, and set the 
    formation energy to ChemPotEntry
    '''
    phased = PhaseDiagram(PDentries)
    chemPotEntries = []
    for entry in PDentries:
        form_E = phased.get_form_energy_per_atom(entry)


        CPentry = ChemPotEntry(entry,form_E, elslist)
#         form_E = entry.energy/entry.composition.get_reduced_formula_and_factor()[1]
# #         print(form_E)
#         CPentry = ChemPotEntry(entry,form_E, elslist)
        chemPotEntries.append(CPentry)
    return chemPotEntries


class ChemPotEntry(MSONable):
    """
    Special entries for chemical potential diagram

    Args:
        entry (ComputedEntry/ComputedStructureEntry/PDEntry/IonEntry): An
            entry object
        formEperatom: formation energy per atom
    """

    def __init__(self, entry,formEperatom, elslist,entry_id=None):
        self.entry = entry
        if entry_id is not None:
            self.entry_id = entry_id
        elif hasattr(entry, "entry_id") and entry.entry_id:
            self.entry_id = entry.entry_id
        else:
            self.entry_id = None
        self.elmNum = len(self.entry.composition.elements)
        self.elmList = self.entry.composition.elements
        ncomp = {}
        self.strelmList = elslist
        for el in elslist:
            if Element(el) not in self.elmList:
                ncomp[str(el)]=0
            else:
                ncomp[str(el)] = self.entry.composition.get_atomic_fraction(el)
        
        self.ncomp = ncomp
        self.form_E = formEperatom
        print(self.name)
        print(self.symbol)
        
    @property
    def name(self):
        return self.entry.name

    @property
    def symbol(self):
        symbol = ""
        pureformula = self.name
        elmap = {i:0 for i in self.strelmList}

        els_seq = []
        for i in Composition(pureformula)._data:
            els_seq.append(str(i))
            elmap[str(i)]=Composition(pureformula)._data[i]
        print(els_seq)
        for el in els_seq:
            if elmap[el] != 0:
                if elmap[el] != 1:
                    if elmap[el] != int(elmap[el]):
                        symbol += '$%s_{%.2f}$' % (el, elmap[el])
                    else:
                        symbol += '$%s_{%d}$' % (el, elmap[el])
                else:
                    symbol += '$%s$' % (el)            
        return symbol
    
    # TODO: not sure if these are strictly necessary with refactor
    def as_dict(self):
        """
        Returns dict which contains Pourbaix Entry data.
        Note that the pH, voltage, H2O factors are always calculated when
        constructing a PourbaixEntry object.
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__}
        d["entry"] = self.entry.as_dict()
        d["entry_id"] = self.entry_id
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Invokes
        """
        entry = PDEntry.from_dict(d["entry"])
        entry_id = d["entry_id"]
        return ChemPotEntry(entry, entry_id)

    @property
    def composition(self):
        """
        Returns composition
        """
        return self.entry.composition

    @property
    def num_atoms(self):
        """
        Return number of atoms in current formula. Useful for normalization
        """
        return self.composition.num_atoms
    
    def normalized_energy_at_conditions(self, udict): #udict = {'Li':-1.3(chemPot),'Ba':-1.6,...}
        newE = self.form_E
        for el in udict:
            if el in self.ncomp:
                newE = newE - self.ncomp[el]*udict[el]
            else:
                newE = newE - 0*udict[el]
    
    
    
    def __repr__(self):
        return "ChemPot Entry : composition = {}, form_E = {}, " \
               "ncomp = {}, entry_id = {} ".format(
            self.entry.composition, self.form_E,
            self.ncomp, self.entry_id)

    def __str__(self):
        return self.__repr__()

class ChemPotDiagram(MSONable):
    """
    Class to create a chemical potential diagram from CPentries

    Args:
        entries (CPentries): Entries list
            containing Solids and Ions or a list of MultiEntries
        elementList: element str list of entries
        fixed:  list of tuples when a/mutiple element(s) chemical 
            potential is fixed at certain value. Eg: [("Fe",-1)]
        limits: boundary limits. 
            Eg: [[-10,0],[-10,0]] for a 2 components system
    """

    def __init__(self, entries,elementList,fixed = None, limits = None):
#         entries = deepcopy(entries)
        self._processed_entries = entries
        self.elementList = elementList
        self._stable_domains, self._stable_domain_vertices, self.volume = \
            self.get_chem_pot_domains(self._processed_entries,elementList,fixed = fixed,limits = limits)
        self.limits = limits
        if fixed != None:
            self.fixedEle = fixed.copy()

    @staticmethod
    def get_chem_pot_domains(CPentries, elementList, fixed = None, phi = None,limits=None):
        """
        Returns a set of CP stable domains (i. e. polygons) in
        mu space from a list of CPentries

        This function works by using scipy's HalfspaceIntersection
        function to construct all of the 2/3-D polygons that form the
        boundaries of the planes corresponding to individual entry
        grand potential energies as a function of chemical potential
        of elements. 

        Args:
            Same as ChemPotDiagram Class

        Returns:
            Returns a dict of the form {entry: [boundary_points]}.
            The list of boundary points are the sides of the N-1
            dim polytope bounding the allowable mu range of each entry.
        """

            
        C=len(elementList)
             
        # Get hyperplanes
        hyperplanes = []
        if fixed == None:
            if limits is None:
                limits = []
                for i in range(len(elementList)):
                    limits += [[-10,0]]
            for entry in CPentries:
                # Create Hyperplanes
                # We're going to make it a variable length
                # The length of the hyperplane will be C + 2, where C = number of components
                # N = number of entries
                # Phi = G - muAxA - muBxB - muCxC
                #  Nx(C+1) matrix        (C+1)x1 matrix    1x1
                # [xA, xB, xC, 1] x [muA; muB; muC; Phi] = [G]
                #  Nx(C+1) matrix        (C+1)x1 matrix    1x1
                # [xA, xB, xC, 1] x [muA; muB; muC; Phi] = [G]
                # FOR HALFSPACEINTERSECTION, the vectors are
                # N x (C+2)
                # [xA, xB, xC, 1, -G]
                
                hyperplane=[]
#                 print(entry.name,entry.ncomp)
                for z in range(0,C):
                    hyperplane.append(entry.ncomp[elementList[z]])
                hyperplane.append(1)
                hyperplane.append(-entry.form_E)
#                 print("hyperplane",hyperplane)
                hyperplanes += [np.array(hyperplane)]
#             print(hyperplanes)
            hyperplanes = np.array(hyperplanes)
            
            print(hyperplanes)
    #         print()
            max_contribs = np.max(np.abs(hyperplanes), axis=0)
#             print(max_contribs)
            g_maxList = [] #limits[0][1], limits[1][1], 0, 1
            for i in range(len(elementList)):
                g_maxList.append(limits[i][1])
            g_maxList.append(0)
            g_maxList.append(1)
            g_max = np.dot(-max_contribs, g_maxList)
#             print()
#             print("g_max",g_max)
#             print(limits)
    
    
            # Add border hyperplanes and generate HalfspaceIntersection
            ## TODO: Now we have to make the border_hyperplanes also N-dimensional. 
            ## You will also have to respecify the variable 'limits' to be N-dimensional. 
            border_hyperplanes = []
            for j in range(2*C):
                border_hyperplane = []
                for i in range(C):
                     
                    if j == 2*i:
                        border_hyperplane.append(-1)
                    elif j == 2*i + 1:
                        border_hyperplane.append(1)
                    else:
                        border_hyperplane.append(0)
                border_hyperplane.append(0)
                if (j%2) == 0:
                     
                    border_hyperplane.append(limits[int(j/2)][0])
                else:
                    border_hyperplane.append(-limits[int((j-1)/2)][1])
                print(border_hyperplane)
                border_hyperplanes.append(border_hyperplane)
            border_hyperplane = []
            for i in range(C):
                border_hyperplane.append(0)
            border_hyperplane.append(-1)
            border_hyperplane.append(2*g_max)
    #         print(border_hyperplane)
            border_hyperplanes.append(border_hyperplane)
    
            print("border_hyperplanes",border_hyperplanes)
            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
#             print(hs_hyperplanes)

            #TODO: This next line, if you need it, should also be made N-dimensional.
            # You can replace X, Y, Z... with X1, X2, X3... Phi, G
#             for ii in hs_hyperplanes:
#                 stringName = "( "
#                 n = 0
#                 for j in elementList:
#                     stringName = stringName + str(ii[n]) + " * mu_" + j +" + "
#                     n +=1
#                 stringName = stringName + str(ii[C]) + " * phi " + "+ " + str(ii[C+1]) +"{-G} <= 0 ) & "
#                 print(stringName)
            #You'll have to make the interior point N-dimensional as well.
            #  I Think if you fix limits to be N-dimensional, the interior point will also be 
            # (N+1)-dimensional, where the +1 is the energy dimension 
            interior_point = np.average(limits, axis=1).tolist() + [g_max]
            print(interior_point)

            hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))
#             print(hs_int.intersections)
            # organize the boundary points by entry
            CPdomains = {entry: [] for entry in CPentries}
            for intersection, facet in zip(hs_int.intersections,
                                           hs_int.dual_facets):
                for v in facet:
                    if v < len(CPentries):
                        this_entry = CPentries[v]
                        CPdomains[this_entry].append(intersection)
    
            # Remove entries with no pourbaix region
#             for k, v in CPdomains.items():
#                 if len(v) ==0:
#                     print(k.name)
#             gaga
            CPdomains = {k: v for k, v in CPdomains.items() if v}
            CP_domain_vertices = {}
            CP_volume = {}
            if phi != None:
                for entry, points in CPdomains.items():
                    points = np.array(points)[:,:C+1]
#                     print('P1',points)

                    points = points[np.lexsort(np.transpose(points))]
#                     print('P2', points)
                    center = np.average(points, axis=0)
                    points_centered = points - center
 
                    point_comparator = lambda x, y: x[0] * y[1] - x[1] * y[0]
                    points_centered = sorted(points_centered,
                                             key=cmp_to_key(point_comparator))
                    points = points_centered + center
#                     print('P3', points)
                    CP_domain_vertices[entry] = points 
                return CP_domain_vertices
            
            for entry, points in CPdomains.items():
#                 print(entry.name)
                points = np.array(points)[:,:C]
                # Initial sort to ensure consistency
                points = points[np.lexsort(np.transpose(points))]

                center = np.average(points, axis=0)
                points_centered = points - center
                
                # Sort points by cross product of centered points,
                # isn't strictly necessary but useful for plotting tools
                point_comparator = lambda x, y: x[0] * y[1] - x[1] * y[0]
                points_centered = sorted(points_centered,
                                         key=cmp_to_key(point_comparator))
                points = points_centered + center

                hull = ConvexHull(points, qhull_options='QJ')
                CP_volume[entry] = hull.volume
                simplices = [Simplex(points[indices]) for indices in hull.simplices]
                CPdomains[entry] = simplices
                CP_domain_vertices[entry] = points

                
        else:
            F = len(fixed)
            fixIndex = [elementList.index(fixed[i][0]) for i in range(F)]
            if limits is None:
                limits = []
                for i in range(C-F):
                    limits += [[-10,0]]
            else:
                newlimits = []
                for i in range(C):
                    if i not in fixIndex:
                        newlimits += [limits[i]]
                limits = newlimits

            for iiii in range(F):
                print("chemical potential of element",fixed[iiii][0]," is fixed at",fixed[iiii][1])

            for entry in CPentries:
                # Create Hyperplanes
                # We're going to make it a variable length
                # The length of the hyperplane will be C + 2 - 1, where C = number of components
                # N = number of entries
                # Phi = G - muAxA - muBxB - muCxC
                # if we fixed muA, then G -muAxA is a constant
                #  NxC matrix        Cx1 matrix    1x1
                # [xB, xC, 1] x [muB; muC; Phi] = [G-muAxA]

                
                # FOR HALFSPACEINTERSECTION, the vectors are
                # N x (C+1)
                # [xB, xC, 1, -G+ sum(all fix muAxA)]
                
                hyperplane=[]
                print(entry.name,entry.ncomp)
                for z in range(0,C):
                    if z not in fixIndex:
                        hyperplane.append(entry.ncomp[elementList[z]])
                hyperplane.append(1)
                formEMultiMux = 0
                for i in fixed:
                    formEMultiMux += i[1]*entry.ncomp[i[0]]
                formEMultiMux = formEMultiMux-entry.form_E
                hyperplane.append(formEMultiMux)
                print("hyperplane",hyperplane)
                hyperplanes += [np.array(hyperplane)]
            hyperplanes = np.array(hyperplanes)
            #########################################################
            C = C-F
            max_contribs = np.max(np.abs(hyperplanes), axis=0)
            print(max_contribs)
            g_maxList = [] #limits[0][1], limits[1][1], 0, 1
            for i in range(C):
                g_maxList.append(limits[i][1])
            g_maxList.append(0)
            g_maxList.append(1)
            g_max = np.dot(-max_contribs, g_maxList)
            print()
            print(g_max)
            print(limits)
    
            # Add border hyperplanes and generate HalfspaceIntersection
            
            ## TODO: Now we have to make the border_hyperplanes also N-dimensional. 
            ## You will also have to respecify the variable 'limits' to be N-dimensional. 
            border_hyperplanes = []
            for j in range(2*C):
                border_hyperplane = []
                for i in range(C):
                     
                    if j == 2*i:
                        border_hyperplane.append(-1)
                    elif j == 2*i + 1:
                        border_hyperplane.append(1)
                    else:
                        border_hyperplane.append(0)
                border_hyperplane.append(0)
                if (j%2) == 0:
                     
                    border_hyperplane.append(limits[int(j/2)][0])
                else:
                    border_hyperplane.append(-limits[int((j-1)/2)][1])
    #             print(border_hyperplane)
                border_hyperplanes.append(border_hyperplane)
            border_hyperplane = []
            for i in range(C):
                border_hyperplane.append(0)
            border_hyperplane.append(-1)
            border_hyperplane.append(2*g_max)
    #         print(border_hyperplane)
            border_hyperplanes.append(border_hyperplane)
    
#             print("border_hyperplanes",border_hyperplanes)
            
            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
#             print(hs_hyperplanes)
    
            #You'll have to make the interior point N-dimensional as well.
            #  I Think if you fix limits to be N-dimensional, the interior point will also be 
            # (N+1)-dimensional, where the +1 is the energy dimension 
            interior_point = np.average(limits, axis=1).tolist() + [g_max]
#             print(interior_point)
    
            hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))
            
            # organize the boundary points by entry
            CPdomains = {entry: [] for entry in CPentries}
            for intersection, facet in zip(hs_int.intersections,
                                           hs_int.dual_facets):
                for v in facet:
                    if v < len(CPentries):
                        this_entry = CPentries[v]
                        CPdomains[this_entry].append(intersection)
    
            # Remove entries with no pourbaix region
            CPdomains = {k: v for k, v in CPdomains.items() if v}
            CP_domain_vertices = {}
            CP_volume = {}
            if len(fixed) == len(elementList)-1:
                for entry, points in CPdomains.items():
#                     points = np.array(points)[:,:C+1]
#                     CPdomains[entry] = points
                    points = np.array(points)[:,:C]
                    CP_domain_vertices[entry] = points
                    CP_volume[entry] = None
            else:
                for entry, points in CPdomains.items():
    #                 if entry.name == 'Ba4NaAl2B8(ClO6)3':
                    points = np.array(points)[:,:C]
                    # Initial sort to ensure consistency
                    points = points[np.lexsort(np.transpose(points))]
        #             print('P2', points)
                    center = np.average(points, axis=0)
                    points_centered = points - center
        
                    # Sort points by cross product of centered points,
                    # isn't strictly necessary but useful for plotting tools
                    '''I do not know if this matters'''
                    ## IF THE FOLLOWING SECTION HAS ERRORS, you can comment out these next 4 lines. 
                    point_comparator = lambda x, y: x[0] * y[1] - x[1] * y[0]
                    points_centered = sorted(points_centered,
                                             key=cmp_to_key(point_comparator))
                    points = points_centered + center
        #             print('P3', points)
                    # Create simplices corresponding to pourbaix boundary
                    hull = ConvexHull(points)
                    CP_volume[entry] = hull.volume
                    simplices = [Simplex(points[indices]) for indices in hull.simplices]
    #                 print(simplices)
                    CPdomains[entry] = simplices
                    CP_domain_vertices[entry] = points
                    

        return CPdomains, CP_domain_vertices, CP_volume

    def find_stable_entry(self, udict):
        """
        Finds stable entry at a pH,V condition
        Args:
            pH (float): pH to find stable entry
            V (float): V to find stable entry
        Returns:
        """
        energies_at_conditions = [e.normalized_energy_at_conditions(udict)
                                  for e in self.stable_entries] #not sure if there is self.stable_entries
        return self.stable_entries[np.argmin(energies_at_conditions)]

#     def get_decomposition_energy(self, entry, pH, V):
#         """
#         Finds decomposition to most stable entry
# 
#         Args:
#             entry (PourbaixEntry): PourbaixEntry corresponding to
#                 compound to find the decomposition for
#             pH (float): pH at which to find the decomposition
#             V (float): voltage at which to find the decomposition
# 
#         Returns:
#             reaction corresponding to the decomposition
#         """
#         # Find representative multientry
#         if self._multielement and not isinstance(entry, MultiEntry):
#             possible_entries = self._generate_multielement_entries(
#                 self._filtered_entries, forced_include=[entry])
# 
#             # Filter to only include materials where the entry is only solid
#             if entry.phase_type == "solid":
#                 possible_entries = [e for e in possible_entries
#                                     if e.phase_type.count("Solid") == 1]
#             possible_energies = [e.normalized_energy_at_conditions(pH, V)
#                                  for e in possible_entries]
#         else:
#             possible_energies = [entry.normalized_energy_at_conditions(pH, V)]
# 
#         min_energy = np.min(possible_energies, axis=0)
# 
#         # Find entry and take the difference
#         hull = self.get_hull_energy(pH, V)
#         return min_energy - hull
# 
#     def get_hull_energy(self, pH, V):
#         all_gs = np.array([e.normalized_energy_at_conditions(
#             pH, V) for e in self.stable_entries])
#         base = np.min(all_gs, axis=0)
#         return base

    @property
    def stable_entries(self):
        """
        Returns the stable entries in the Pourbaix diagram.
        """
        return list(self._stable_domains.keys())

    @property
    def unstable_entries(self):
        """
        Returns all unstable entries in the Pourbaix diagram
        """
        return [e for e in self.all_entries if e not in self.stable_entries]

    @property
    def all_entries(self):
        """
        Return all entries used to generate the pourbaix diagram
        """
        return self._processed_entries

    @property
    def unprocessed_entries(self):
        """
        Return unprocessed entries
        """
        return self._unprocessed_entries

    def as_dict(self, include_unprocessed_entries=False):
        if include_unprocessed_entries:
            entries = [e.as_dict() for e in self._unprocessed_entries]
        else:
            entries = [e.as_dict() for e in self._processed_entries]
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "entries": entries,
             "comp_dict": self._elt_comp,
             "conc_dict": self._conc_dict}
        return d

    @classmethod
    def from_dict(cls, d):
        decoded_entries = MontyDecoder().process_decoded(d['entries'])
        return cls(decoded_entries, d.get('comp_dict'),
                   d.get('conc_dict'))


class ChemPotPlotter:
    """
    A plotter class for phase diagrams.

    Args:
        phasediagram: A PhaseDiagram object.
        show_unstable: Whether unstable phases will be plotted as well as
            red crosses. Defaults to False.
    """

    def __init__(self, ChemPotDiagram, elementList = None, fixed = None):
        self._cp = ChemPotDiagram
        
        if elementList == None:
            self.elementList = ChemPotDiagram.elementList
        else:
            self.elementList = elementList
            
        if fixed == None:
            self.sliceTitle = False
        else:
            self.fixed = fixed
            self.sliceTitle = True
    def show(self, *args, **kwargs):
        """
        Shows the pourbaix plot

        Args:
            *args: args to get_pourbaix_plot
            **kwargs: kwargs to get_pourbaix_plot

        Returns:
            None
        """
        plt = self.get_chempot_plot(*args, **kwargs)
#         plt.savefig('BaMnN'+"_ChemPot",dpi = 150)
        plt.show()
        
        
    def get_polar_chart_plot(self, compound,elementlist,fixedEle,limits=None):
        eql = EquilLine(self._cp._processed_entries,elementlist,fixed = fixedEle, limits = limits)
        elsnum = len(self.elementList)

        fig = go.Figure()
        for entry,vertices in eql._stable_domain_vertices.items():
            if entry.name == compound:
                for el in self.elementList:
#                     c = next(color)
                    index = self.elementList.index(el)
                    data = vertices[:,index:index+1]
                    print(min(data),max(data))
                    fig.add_trace(go.Scatterpolar(
                            r = [float(min(data)),float(max(data))],
                            theta = [index*360/elsnum, index*360/elsnum],
                            line = dict(width = 15),
                            mode = 'lines+text',
                            name = el,
                            opacity = 0.5
                        ))
        tvList = [i*360/elsnum for i in range(elsnum)]
        strmu = 'Chem Pot '
        ttList = [strmu + i for i in self.elementList]


        fig.update_layout(
            template=None,
            title = compound + " chemical potential polar chart " + str(fixedEle),
            titlefont = dict(size = 15),
            font=dict(size=20, family='Arial', color='black'),
            polar = dict(
                radialaxis = dict(range = [-10,0], showticklabels=True,),
                angularaxis = dict(
                tickmode = 'array',
                tickvals = tvList,
                ticktext = ttList
            ) )  )        
        fig.show()
                    
        if elsnum == 2:
            fig = go.Figure()
            for e, po in eql._stable_domain_vertices.items():
                if e.name == compound:
                    print(po)
                    points = po
            fig.add_traces(go.Scatter(x = [points[0][0],points[1][0]], 
                                      y = [points[0][1],points[1][1]],
                                      mode='lines',
                                      line=dict( width=2),))
            fig.update_xaxes(title_text="Chemical Potential " + self.elementList[0])
            fig.update_yaxes(title_text="Chemical Potential " + self.elementList[1])
#             fig.update_layout(template = None)
#             fig.update_xaxes(title_font=dict(size=20, family='Calibri', color='black'),showgrid=False,zeroline=False,showline=True, linewidth=2, linecolor='black',mirror=True)
#             fig.update_yaxes(title_font=dict(size=20, family='Calibri', color='black'),showgrid=False, zeroline=False,showline=True, linewidth=2, linecolor='black',mirror=True)
            fig.show()
        
        
        
    def get_phimumu_plot(self,alpha = 0.2,edc = None,limits = None,show_label=True, label_equiline=True):
        CP_domain_vertices = self._cp._stable_domain_vertices
        CP_domain_vertices2 = self._cp.get_chem_pot_domains(
            self._cp._processed_entries,self._cp.elementList,phi = 1,limits = limits[:2])
        n=len(CP_domain_vertices.keys())
        if limits == None:
            limits = [[-10,0],[-10,0],[-10,10]]
        jet= plt.get_cmap('rainbow')
        color=iter(jet(np.linspace(0,1,n)))
        fig = plt.figure(figsize=(9.2, 7))
        ax = a3.Axes3D(fig)
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        cpentries = list(CP_domain_vertices.keys())
        for e in cpentries:
            print(e.name)
        cpentries = sorted(cpentries,key = lambda e:e.entry.composition.get_atomic_fraction(self.elementList[0]))

        eql = EquilLine(self._cp._processed_entries,self.elementList,fixed = None, limits = limits[:2])
        colordict = {}
        for e in cpentries:
            print(e.name)
        colordict = {'O2': [0.5, 0.0, 1.0, 1.0], 'MnO2': [0.09999999999999998, 0.5877852522924731, 0.9510565162951535, 1.0], 'Mn2O3': [0.30000000000000004, 0.9510565162951535, 0.8090169943749475, 1.0], 'Mn3O4': [0.7, 0.9510565162951536, 0.5877852522924731, 1.0], 'MnO': [1.0, 0.5877852522924732, 0.30901699437494745, 1.0], 'Mn': [1.0, 1.2246467991473532e-16, 6.123233995736766e-17, 1.0]}


        for key in cpentries:
#             print(key.name, CP_domain_vertices[key])
            newCPver = []
            for e in CP_domain_vertices[key]:
                e= np.append(e,limits[-1][0])
                newCPver.append(e)
            newCPver = np.array(newCPver)
            CP_domain_vertices[key] = newCPver
            newCPver = np.append(newCPver,[newCPver[0]],axis = 0)

            c=next(color)
#             colordict[key.name] = list(c)
            a = alpha
            
            pc = a3.art3d.Poly3DCollection([newCPver], alpha = a, facecolor=colordict[key.name],edgecolor = edc)
            ax.add_collection3d(pc)
            center = np.average(CP_domain_vertices[key], axis=0)
            if label_equiline:
                ax.text(center[0],center[1],center[2],key.name,ha='center',
                        va='center',**text_font,color = 'k')

            pc = a3.art3d.Poly3DCollection([CP_domain_vertices2[key]], 
                                           alpha = a, facecolor=colordict[key.name],edgecolor = edc)
            ax.add_collection3d(pc)
            center = np.average(CP_domain_vertices2[key], axis=0)
#             if key.name=="Mn3O4":
#                 ax.scatter3D(center[0],center[1],center[2],
#                              marker = "o",s = 100, color = color)

            if label_equiline:
                ax.text(center[0],center[1],center[2],key.name,ha='center',
                        va='center',**text_font,color = 'k')
        
#         for a in colordict:
#             print(a, colordict[a])
#         colordict = {a: list(colordict[a]) for a in colordict}
#         print(colordict)
        
        for entry, vertices in eql._stable_domain_vertices.items():
            zver = np.array([0 for i in range(len(vertices[:,:1]))])
            zver1 = np.array([limits[-1][0] for i in range(len(vertices[:,:1]))])
            print(zver)
            ax.plot(vertices[:,:1].transpose()[0],vertices[:,1:2].transpose()[0],
                    zver,linewidth = 2.25,color = colordict[entry.name])
            ax.plot(vertices[:,:1].transpose()[0],vertices[:,1:2].transpose()[0],
                    zver1,linewidth = 2.25,color = colordict[entry.name])
        ax.grid(b=None)
        ax.dist=10
        ax.azim=30
        ax.elev=10
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.invert_yaxis()
        limits.append([-2,2])
        ax.set_zlim(limits[2])
        ax.set_zticks(range(-12, 9, 4))
        if show_label:
            print()
            ax.set_xlabel('Chem Pot '+self.elementList[0],fontname='Arial',fontsize = 15)
            ax.set_ylabel('Chem Pot '+self.elementList[1],fontname='Arial',fontsize = 15)
            ax.set_zlabel('Grand Pot Φ',fontname='Arial',fontsize = 15)
            ax.set_xticks(np.arange(limits[0][0],limits[0][1]+0.01,step = 2))
            ax.set_yticks(np.arange(limits[1][0],limits[1][1]+0.01,step = 2))
            ax.set_zticks(np.arange(limits[2][0],limits[2][1]+0.01,step = 2))
        else:
            ax.set_xticks(np.arange(limits[0][0],limits[0][1]+0.01,step = 2))
            ax.set_yticks(np.arange(limits[1][0],limits[1][1]+0.01,step = 2))
            ax.set_zticks(np.arange(limits[2][0],limits[2][1]+0.01,step = 2))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        plt.show()


    def get_ON_ternary_projection_equilLine_plot(self,projEle, texts,colordict=None, fig = None,compound = None,
                            label_domains=False,edc = None, 
                            limits = None,alpha = 0.2,show_polytope = True,
                            label_equilLine = True):
        eql = EquilLine(self._cp._processed_entries,self.elementList,fixed = None, limits = limits)

        projIndex = [self.elementList.index(e) for e in projEle if e in self.elementList]
        F = len(projIndex)
        C = len(self.elementList)
        if limits == None:
            limits = []
            for i in range(F):
                limits += [[-10,0]]
        else:
            newlimits = []
            for stre in projEle:
                i = self.elementList.index(stre)
                newlimits += [limits[i]]
            limits = newlimits
            print(limits)
        label_font = {'fontname':'Calibri', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Calibri', 'size':'12', 'weight':'normal'}
        if len(projEle) == 2:
            if fig ==None:
                fig = plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.set_xlim([-6,2])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                           colors='k',grid_color='k', grid_alpha=0.5)
            ax.set_ylim([-8,2])

            if show_polytope:
                for entry, vertices1 in self._cp._stable_domain_vertices.items():
#                     if entry.name == "CaVF6":
                    dataList = [vertices1[:,i:i+1] for i in projIndex]
                    data = np.hstack([dataList[0],dataList[1]])
                    hull = ConvexHull(data)
                    vertices = [data[s] for s in hull.vertices]
    #                 print(entry.name,vertices)
                    center = np.average(vertices, axis=0)
                    x, y = np.transpose(np.vstack([vertices, vertices[0]]))
    #                 print(np.vstack([vertices, vertices[0]]))
    #                 print(np.transpose(np.vstack([vertices, vertices[0]])))
                    plt.fill(x, y, alpha = 0.1,
                             facecolor = colordict[entry.name],edgecolor = edc)
    
                    if label_domains == True:
                        plt.annotate(entry.name, center, ha='center',
                                     va='center', **text_font,
                                     color = colordict[entry.name])
                
            plt.xlabel('Chem Pot '+projEle[0],**label_font)
            plt.ylabel('Chem Pot '+projEle[1],**label_font)
            for entry, vertices in eql._stable_domain_vertices.items():
                if entry.name == compound:

                    dataList = [vertices[:,i:i+1] for i in projIndex]
                    data = np.hstack([dataList[0],dataList[1]])
                    print(entry.name)
                    print(data.tolist())
                    hull = ConvexHull(data, qhull_options='QJ')
                    points = [data[s] for s in hull.vertices]
                    
                    center = np.average(points, axis=0)
                    points_centered = points - center
                    # Sort points by cross product of centered points,
                    # isn't strictly necessary but useful for plotting tools 
                    point_comparator = lambda x, y: x[0] * y[1] - x[1] * y[0]
                    points_centered = sorted(points_centered,
                                             key=cmp_to_key(point_comparator))
                    points = points_centered + center
                    x, y = np.transpose(np.vstack([points, points[0]]))
    
                    plt.fill(x, y, alpha = alpha,facecolor = colordict[entry.name],edgecolor = edc)
                    if label_equilLine:
                        text = plt.annotate(entry.name, center, ha='center',
                                     va='center', **text_font,
                                     color = colordict[entry.name])
                        texts.append(text)
                        adjust_text(texts)
#             plt.show()
        if len(projEle) == 3:
            jet= plt.get_cmap('gist_rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            colordict = {}
            for e in self._cp._stable_domain_vertices:
                c = next(color)
                colordict[e.name] = list(c)
            
            '''set color based on Li atomic fraction'''
#             entries_names = []
#             for e in self._cp._stable_domain_vertices:
#                 stree = [str(ell) for ell in e.composition.elements]
#                 if "Li" in stree:
#                     entries_names.append(e.name)
#             entries_names.sort(key= lambda x: Composition(x).get_atomic_fraction("Li"), reverse=True)
#             jet= plt.get_cmap('plasma')
#             n = len(entries_names)
#             color=iter(jet(np.linspace(0,1,n)))
#             for e in entries_names:
#                 c = next(color)
#                 colordict[e] = list(c)
            
            if show_polytope:
                for e in self._cp._stable_domain_vertices:
#                     if e.name == "Ca":
                    data = self._cp._stable_domain_vertices[e]

                    dataList = [data[:,i:i+1] for i in projIndex]
                    data = np.hstack([dataList[0],dataList[1],dataList[2]])
                    hull = ConvexHull(data)
                    simplices = hull.simplices 
                    org_triangles = [data[s] for s in simplices]
                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = 0.1, facecolor=colordict[e.name],edgecolor = edc)
                    ax.add_collection3d(pc)
#                     print(vertices)
                    if label_domains == True:
                        
                        
                        vertices = [data[s] for s in hull.vertices]
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],e.name,ha='center',
                                va='center',**text_font,color = colordict[e.name])
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim([limits[0][0]-0.5,limits[0][1]+0.5])
            ax.set_ylim([limits[1][0]-0.5,limits[1][1]+0.5])
            ax.set_zlim([limits[2][0]-0.5,limits[2][1]+0.5])
            ax.set_xlabel('X: Chem Pot '+projEle[0],fontname='Calibri',fontsize = 12)
            ax.set_ylabel('Y: Chem Pot '+projEle[1],fontname='Calibri',fontsize = 12)
            ax.set_zlabel('Z: Chem Pot '+projEle[2],fontname='Calibri',fontsize = 12)
#             hull = ConvexHull(eql.intersection)
#             simplices = hull.simplices
            centers = {}
            for entry, vertices in eql._stable_domain_vertices.items():
                tentries = ['Li2O', 'O2', 'ClO3', 'LiClO4', 'Li2O2', 'ClO2', 'LiO8', 'LiCl', 'Li', 'Cl2', 'Cl2O7', 'Cl2O']
                if entry.name not in tentries:
#                     if len(entry.composition.elements) < 5:
                    continue
                dataList = [vertices[:,i:i+1] for i in projIndex]
                data = np.hstack([dataList[0],dataList[1],dataList[2]])
                hull = ConvexHull(data, qhull_options='QJ')
                simplices = hull.simplices 
                org_triangles = [data[s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, \
                     alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_equilLine:
                    cvertices = [data[s] for s in hull.vertices]
                    center = np.average(cvertices, axis=0)
                    centers[entry.name] = list(center)
                    ax.text(center[0],center[1],center[2],entry.name,ha='center',
                            va='center',**text_font,color = 'k')
        projStr = ''
#         print(colordict)
#         print(centers)
        
        for ee in projEle:
            projStr += ee + " "
        if compound != None:
            plt.title('Projection of '+compound + " system to "+ projStr)
        else:
            eleStr = "( "
            for ii in self.elementList:
                eleStr += ii +" "
            eleStr += ")" 
            plt.title('Projection of '+ eleStr + " system to "+ projStr,**text_font)
        return fig
    
    def get_mu_xx_plot_reverse(self,projEle, compound = None,
                            label_domains=False,edc = None, 
                            limits = None,alpha = 0.2,show_polytope = True,
                            label_equilLine = True, label_phase2=False, label_phase3=False):
        eql = EquilLine(self._cp._processed_entries,self.elementList,fixed = None, limits = limits)

        projIndex = [self.elementList.index(e) for e in projEle if e in self.elementList]
        nonProjIndex = [self.elementList.index(e) for e in self.elementList if e not in projEle]


        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(projEle) == 1:

            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            print(projIndex)
            ax.set_ylim(limits[projIndex[0]])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                           colors='k',grid_color='k', grid_alpha=0.5)
            ax.set_xlim([-0.002,1.005])
#             ax.set_ylim([-12,2])
            jet= plt.get_cmap('rainbow')
            n=len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            colordict = {}
            for e in self._cp._stable_domain_vertices:
                c = next(color)
                colordict[e.name] = c.tolist()
            colordict1 = {'MnO': [0.5, 0.0, 1.0, 1.0], 'MnO2': [0.37450980392156863, 0.19584546700716696, 0.9951469164070644, 1.0], 'Li9Mn20O40': [0.24901960784313726, 0.38410574917192586, 0.9806347704689777, 1.0], 'Li2MnO3': [0.12352941176470589, 0.5574894393428855, 0.9566044195004408, 1.0], 'Mn3O4': [0.0019607843137254832, 0.7092813076058535, 0.9232891061054894, 1.0], 'Li6MnO4': [0.12745098039215685, 0.8336023852211195, 0.8810121942857845, 1.0], 'O2': [0.2529411764705882, 0.9256376597815562, 0.8301840308155507, 1.0], 'Li2O': [0.3784313725490196, 0.9818225628535369, 0.7712979623471807, 1.0], 'Mn2O3': [0.503921568627451, 0.9999810273487268, 0.7049255469061472, 1.0], 'Li4Mn5O12': [0.6294117647058823, 0.9794097676013659, 0.631711006253251, 1.0], 'Mn': [0.7549019607843137, 0.9209055179449537, 0.5523649729605058, 1.0], 'LiMn2O4': [0.8803921568627451, 0.8267341748257635, 0.4676575928925868, 1.0], 'Li2O2': [1.0, 0.7005430375932911, 0.37841105004231035, 1.0], 'LiO8': [1.0, 0.5472195469221114, 0.28549158627534216, 1.0], 'Li9Mn14O32': [1.0, 0.37270199199091436, 0.18980109344182594, 1.0], 'Li11Mn13O32': [1.0, 0.18374951781657037, 0.09226835946330202, 1.0], 'LiMnO2': [1.0, 1.2246467991473532e-16, 6.123233995736766e-17, 1.0]}
            for i in colordict1:
                colordict[i] = np.array(colordict1[i])
            ax.set_ylabel('Chem Pot '+projEle[0],**label_font)
            ax.set_xlabel("{el1}x{el2}(1-x)".format(el1=self.elementList[nonProjIndex[0]], 
                                                    el2=self.elementList[nonProjIndex[1]]),**label_font)

            hull = ConvexHull(eql.intersection)
            simplices = hull.simplices
            org_triangles = [eql.intersection[s] for s in simplices]
            phase1 = Phase1(eql, org_triangles, projIndex, nonProjIndex)

            '''plot 2-phase coexistence'''

            twoPhases = []
            for entry1, entry2 in list(combinations(eql._processed_entries,2)):
                for (line1,line2) in itertools.product(phase1.linesDict[entry1], phase1.linesDict[entry2]):
                    if compare_xy_lines(line1,line2):
                        print(entry1.name,entry2.name)

                        x1 = phase1.get_comp_after_fix_mu(entry1)
                        x2 = phase1.get_comp_after_fix_mu(entry2)
                        if x1==None or x2==None:
                            continue
                        phase2 = Phase23(triangles = [[[line1[0][projIndex[0]],x1],[line1[1][projIndex[0]],x1],[line1[0][projIndex[0]],x2]],
                                                      [[line1[1][projIndex[0]],x2],[line1[1][projIndex[0]],x1],[line1[0][projIndex[0]],x2]]],
                               names=[entry1.name,entry2.name],
                               color=(colordict[entry1.name] + colordict[entry2.name])/2
                               )
                        twoPhases.append(phase2)
                        break
            phase2colordict = {}
            for phase2 in twoPhases:
                patches = []
                for triangle in phase2.triangles:
                    triangle = [[k[1], k[0]] for k in triangle]
                    polygon = Polygon(triangle, True)
                    patches.append(polygon)
                pc = PatchCollection(patches, alpha = alpha,
                                           facecolor = phase2.color)#,edgecolor = 'w')
                ax.add_collection(pc)
#                 print("-".join(phase2.names),phase2.color.tolist())
                phase2colordict["-".join(sorted(phase2.names))] = phase2.color.tolist()
                if label_phase2:
                    center = np.average(phase2.vertices, axis=0)
                    ax.text(center[1],center[0],"-".join(phase2.names),c='k',ha='center',
                            va='center',**text_font)
#             print(colordict["LiMn2O4"].tolist())
#             print(phase2colordict)
            '''plot a line, single phase'''
            for entry, vertices in eql._stable_domain_vertices.items():
                print(entry.name)
                if len(entry.composition.elements) == 1 and str(entry.composition.elements[0]) == projEle[0]:
                    continue
                mu_interest = vertices[:,projIndex[0]:projIndex[0]+1]
                print([max(mu_interest), min(mu_interest)])
                x1 = phase1.get_comp_after_fix_mu(entry)
                plt.plot([x1,x1], [max(mu_interest), min(mu_interest)],
                         linewidth = 4,color = colordict[entry.name])
                if label_domains:
                    plt.annotate(entry.symbol, 
                                 [x1, (max(mu_interest)+min(mu_interest))/2], ha='center',
                                 va='center', **text_font,color = colordict[entry.name])
                
            threePhases = []
#             for i in phase1.vertices:
#                 print(i.name)
#             print()
#             for i in eql._processed_entries:
#                 print(i.name)
            for entry1, entry2, entry3 in list(combinations(phase1.vertices.keys(),3)):
                for (v1,v2,v3) in itertools.product(phase1.vertices[entry1], phase1.vertices[entry2], phase1.vertices[entry3]):
                    if compare_xy_vertices(v1.tolist(),v2.tolist(),v3.tolist(),n=3):

                        x1 = phase1.get_comp_after_fix_mu(entry1)
                        x2 = phase1.get_comp_after_fix_mu(entry2)
                        x3 = phase1.get_comp_after_fix_mu(entry3)
                        if x1==None or x2==None or x3==None:
                            continue
                        v1= [v1[projIndex[0]], x1]
                        v2= [v2[projIndex[0]], x2]
                        v3= [v3[projIndex[0]], x3]

                        phase3 = Phase23(triangles = [[v1,v2,v3]],
                               names=[entry1.name,entry2.name,entry3.name]
                               )
                        threePhases.append(phase3)
                        break
            for phase3 in threePhases:
                patches = []
                for triangle in phase3.triangles:
                    triangle = [[k[1], k[0]] for k in triangle]
                    polygon = Polygon(triangle, True)
                    patches.append(polygon)
                pc = PatchCollection(patches, alpha = 0.4,
                                           facecolor = "r",edgecolor = 'r',linewidths=1)
                ax.add_collection(pc)
                if label_phase3:
                    center = np.average(phase3.vertices, axis=0)
                    ax.text(center[1],center[0],"-".join(phase3.names),c='k',ha='center',
                            va='center',**text_font)    
        plt.show() 
                

        


    def get_mu_xx_plot(self,projEle, compound = None,
                            label_domains=False,edc = None, 
                            limits = None,alpha = 0.2,show_polytope = True,
                            label_equilLine = True, label_phase2=False, label_phase3=False):
        eql = EquilLine(self._cp._processed_entries,self.elementList,fixed = None, limits = limits)

        projIndex = [self.elementList.index(e) for e in projEle if e in self.elementList]
        nonProjIndex = [self.elementList.index(e) for e in self.elementList if e not in projEle]
        

        label_font = {'fontname':'Calibri', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Calibri', 'size':'15', 'weight':'normal'}
        if len(projEle) == 1:

            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            print(projIndex)
            ax.set_xlim(limits[projIndex[0]])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                           colors='k',grid_color='k', grid_alpha=0.5)
            ax.set_ylim([-0.002,1.005])
#             ax.set_ylim([-12,2])
            jet= plt.get_cmap('rainbow')
            n=len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            colordict = {}
            for e in self._cp._stable_domain_vertices:
                c = next(color)
                colordict[e.name] = c
            ax.set_xlabel('Chem Pot '+projEle[0],**label_font)
            ax.set_ylabel("{el1}x{el2}(1-x)".format(el1=self.elementList[nonProjIndex[0]], 
                                                    el2=self.elementList[nonProjIndex[1]]),**label_font)

            hull = ConvexHull(eql.intersection)
            simplices = hull.simplices
            org_triangles = [eql.intersection[s] for s in simplices]
            phase1 = Phase1(eql, org_triangles, projIndex, nonProjIndex)

            '''plot 2-phase coexistence'''

            twoPhases = []
            for entry1, entry2 in list(combinations(eql._processed_entries,2)):
                for (line1,line2) in itertools.product(phase1.linesDict[entry1], phase1.linesDict[entry2]):
                    if compare_xy_lines(line1,line2):
                        print(entry1.name,entry2.name)

                        x1 = phase1.get_comp_after_fix_mu(entry1)
                        x2 = phase1.get_comp_after_fix_mu(entry2)
                        if x1==None or x2==None:
                            continue
                        phase2 = Phase23(triangles = [[[line1[0][projIndex[0]],x1],[line1[1][projIndex[0]],x1],[line1[0][projIndex[0]],x2]],
                                                      [[line1[1][projIndex[0]],x2],[line1[1][projIndex[0]],x1],[line1[0][projIndex[0]],x2]]],
                               names=[entry1.name,entry2.name],
                               color=(colordict[entry1.name] + colordict[entry2.name])/2
                               )
                        twoPhases.append(phase2)
                        break
            for phase2 in twoPhases:
                patches = []
                for triangle in phase2.triangles:
                    polygon = Polygon(triangle, True)
                    patches.append(polygon)
                pc = PatchCollection(patches, alpha = alpha,
                                           facecolor = phase2.color)#,edgecolor = 'w')
                ax.add_collection(pc)
                print("-".join(phase2.names),phase2.color.tolist())

                if label_phase2:
                    center = np.average(phase2.vertices, axis=0)
                    ax.text(center[0],center[1],"-".join(phase2.names),c='k',ha='center',
                            va='center',**text_font)

            '''plot a line, single phase'''
            for entry, vertices in eql._stable_domain_vertices.items():
                print(entry.name)
                if len(entry.composition.elements) == 1 and str(entry.composition.elements[0]) == projEle[0]:
                    continue
                mu_interest = vertices[:,projIndex[0]:projIndex[0]+1]
                print([max(mu_interest), min(mu_interest)])
                x1 = phase1.get_comp_after_fix_mu(entry)
                plt.plot([max(mu_interest), min(mu_interest)],
                         [x1,x1],linewidth = 3,color = colordict[entry.name])
                if label_domains:
                    plt.annotate(generate_entry_label(entry), 
                                 [(max(mu_interest)+min(mu_interest))/2, x1], ha='center',
                                 va='center', **text_font,color = colordict[entry.name])
                
            threePhases = []
            for entry1, entry2, entry3 in list(combinations(eql._processed_entries,3)):
                for (v1,v2,v3) in itertools.product(phase1.vertices[entry1], phase1.vertices[entry2], phase1.vertices[entry3]):
                    if compare_xy_vertices(v1.tolist(),v2.tolist(),v3.tolist(),n=3):

                        x1 = phase1.get_comp_after_fix_mu(entry1)
                        x2 = phase1.get_comp_after_fix_mu(entry2)
                        x3 = phase1.get_comp_after_fix_mu(entry3)
                        if x1==None or x2==None or x3==None:
                            continue
                        v1= [v1[projIndex[0]], x1]
                        v2= [v2[projIndex[0]], x2]
                        v3= [v3[projIndex[0]], x3]

                        phase3 = Phase23(triangles = [[v1,v2,v3]],
                               names=[entry1.name,entry2.name,entry3.name]
                               )
                        threePhases.append(phase3)
                        break
            for phase3 in threePhases:
                patches = []
                for triangle in phase3.triangles:
                    polygon = Polygon(triangle, True)
                    patches.append(polygon)
                pc = PatchCollection(patches, alpha = 0.4,
                                           facecolor = "r",edgecolor = 'r',linewidths=1)
                ax.add_collection(pc)
                if label_phase3:
                    center = np.average(phase3.vertices, axis=0)
                    ax.text(center[0],center[1],"-".join(phase3.names),c='k',ha='center',
                            va='center',**text_font)    


        plt.show() 

    def get_projection_equilLine_mu_x_plot(self,projEle, compound = None,
                            label_domains=False,edc = None, 
                            limits = None,alpha = 0.2,show_polytope = True,
                            label_equilLine = True):

        eql = EquilLine(self._cp._processed_entries,self.elementList,fixed = None, limits = limits)

        projIndex = [self.elementList.index(e) for e in projEle if e in self.elementList]
        nonProjIndex = [self.elementList.index(e) for e in self.elementList if e not in projEle]

        F = len(projIndex)
        C = len(self.elementList)
        if limits == None:
            limits = []
            for i in range(F):
                limits += [[-10,0]]
        else:
            newlimits = []
            for stre in projEle:
                i = self.elementList.index(stre)
                newlimits += [limits[i]]
            limits = newlimits
            print(limits)
        label_font = {'fontname':'Calibri', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Calibri', 'size':'15', 'weight':'normal'}
        if len(projEle) == 2:

            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.set_xlim(limits[0])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                           colors='k',grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])
#             ax.set_ylim([-12,2])
            jet= plt.get_cmap('rainbow')
            n=len(eql._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            colordict = {}
            for e in eql._stable_domain_vertices:
                c = next(color)
                colordict[e.name] = c

            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.grid(b=None)
            ax.set_xlim([limits[0][0]-0.5,limits[0][1]+0.5])
            ax.set_ylim([limits[1][0]-0.5,limits[1][1]+0.5])
            ax.set_zlim([0,1])
#             ax.set_xlabel('Chem Pot '+projEle[0],**label_font)
#             ax.set_ylabel('Chem Pot '+projEle[1],**label_font)
#             ax.set_zlabel('Composition '+self.elementList[nonProjIndex[0]],**label_font)
            
            hull = ConvexHull(eql.intersection)
            simplices = hull.simplices
            org_triangles = [eql.intersection[s] for s in simplices]
            '''tridict has the triangle of each phase to plot in mu-mu-x space
               new_vertices has the vertices of each phase in mu-mu-x space'''
            new_vertices = {entry:[] for entry in eql._processed_entries}
            triDict = {entry: [] for entry in eql._processed_entries}

            for entry, vertices in eql._stable_domain_vertices.items():

                for tri in org_triangles:
                    tri1 = []
                    if all(x in vertices.tolist() for x in tri.tolist()):
                        for ver in tri:
                            ver[nonProjIndex[0]]=Composition(entry.name).get_atomic_fraction(self.elementList[nonProjIndex[0]])
                            ver1=[ver[projIndex[0]],ver[projIndex[1]],ver[nonProjIndex[0]]]
                            if ver1 not in new_vertices[entry]:
                                new_vertices[entry].append(ver1)
                            tri1.append(ver1)
                        triDict[entry].append(tri1)

            triDict = {k: v for k, v in triDict.items() if v}
            '''print edge, linesDict has all boundary edges of each phase'''
            linesDict = {entry: [] for entry in triDict}
            for entry in triDict:

#                 if entry.name == "BaN2":
                lines = []
                for tri in triDict[entry]:
                    lines += list(combinations(tri,2))
                lines1 = []
                for i in lines:
                    if i not in lines1:
                        lines1.append(i)
                    else:
                        lines1.remove(i)
                linesDict[entry] = lines1
                print()
                print(entry.name, len(lines1))
                for ii in lines1:
                    print(ii)
                for li in lines1:
                    li = np.array(li)
                    ax.plot(li[:,0],li[:,1],li[:,2],color="k",alpha = 0.5,linewidth=2)
            '''plot eql face, single phase'''
            for entry, vertices in eql._stable_domain_vertices.items():
  
                pc = a3.art3d.Poly3DCollection(triDict[entry], alpha = alpha,
                                               facecolor = colordict[entry.name])#,edgecolor = 'w')
                ax.add_collection3d(pc)
                print(entry.name)
                if label_equilLine:
                    center = np.average(new_vertices[entry], axis=0)
                    if len(entry.composition.elements)==3:
                        continue
                    ax.text(center[0],center[1],center[2],entry.symbol,c='k',ha='center',
                            va='center',**text_font)
#             gaga
            '''plot 2-phase coexistence'''
            for a,b in linesDict.items():
                print(a.name, b)
            twoPhases = []
            for entry1, entry2 in list(combinations(eql._stable_domain_vertices.keys(),2)):
                for (line1,line2) in itertools.product(linesDict[entry1], linesDict[entry2]):
                    if compare_xy_lines(line1,line2):
                        print(entry1.name,entry2.name)
                        phase2 = Phase23(triangles = [[line1[0],line1[1],line2[0]], [line2[0],line2[1],line1[1]]],
                               names=[entry1.name,entry2.name],
                               color=(colordict[entry1.name] + colordict[entry2.name])/2
                               )
                        twoPhases.append(phase2)
                        break
            for phase2 in twoPhases:
  
                pc = a3.art3d.Poly3DCollection(phase2.triangles, alpha = alpha,
                                               facecolor = phase2.color)#,edgecolor = 'w')
                ax.add_collection3d(pc)
#                 if label_equilLine:
#                     center = np.average(phase2.vertices, axis=0)
#                     ax.text(center[0],center[1],center[2],"-".join(phase2.names),c='k',ha='center',
#                             va='center',**text_font)
            threePhases = []
            for entry1, entry2, entry3 in list(combinations(eql._processed_entries,3)):
                for (v1,v2,v3) in itertools.product(new_vertices[entry1], new_vertices[entry2], new_vertices[entry3]):
                    if compare_xy_vertices(v1,v2,v3):
                        print(entry1.name,entry2.name, entry3.name)
                        phase3 = Phase23(triangles = [[v1,v2,v3]],
                               names=[entry1.name,entry2.name,entry3.name]
                               )
                        threePhases.append(phase3)
                        break
            for phase3 in threePhases:
  
                pc = a3.art3d.Poly3DCollection(phase3.triangles, alpha = 0.4,
                                               facecolor = "red",edgecolor = 'r',linewidths=1)
                ax.add_collection3d(pc)
#                 if label_equilLine:
#                     center = np.average(phase3.vertices, axis=0)
#                     ax.text(center[0],center[1],center[2],"-".join(phase3.names),c='k',ha='center',
#                             va='center',**text_font)    


        plt.show() 
        # return plt

    def get_projection_equilLine_plot(self, projEle, compound = None,
                            label_domains=False,edc = None, 
                            limits = None,alpha = 0.2,show_polytope = True,
                            label_equilLine = True, bold_boundary = False,
                            title = True):
        eql = EquilLine(self._cp._processed_entries,self.elementList,fixed = None, limits = limits)

        projIndex = [self.elementList.index(e) for e in projEle if e in self.elementList]
        F = len(projIndex)
        C = len(self.elementList)
        if limits == None:
            limits = []
            for i in range(F):
                limits += [[-10,0]]
        else:
            newlimits = []
            for stre in projEle:
                i = self.elementList.index(stre)
                newlimits += [limits[i]]
            limits = newlimits
            print(limits)
        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        
        if len(projEle) == 2:

            fig = plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                           colors='k',grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])
            ax.set_xlim(limits[0])
            ax.set_ylim([-5,2])
            ax.set_xlim([-7,0])
#             ax.set_ylim([-12,2])
            jet= plt.get_cmap('rainbow')
            n=len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            colordict = {}
            for e in self._cp._stable_domain_vertices:
                c = next(color)
                colordict[e.name] = c
            if show_polytope:
                for entry, vertices1 in self._cp._stable_domain_vertices.items():
                    dataList = [vertices1[:,i:i+1] for i in projIndex]
                    data = np.hstack([dataList[0],dataList[1]])
                    hull = ConvexHull(data)
                    vertices = [data[s] for s in hull.vertices]
    #                 print(entry.name,vertices)
                    center = np.average(vertices, axis=0)
                    x, y = np.transpose(np.vstack([vertices, vertices[0]]))
    #                 print(np.vstack([vertices, vertices[0]]))
    #                 print(np.transpose(np.vstack([vertices, vertices[0]])))
                    plt.fill(x, y, alpha = 0.1,
                             facecolor = colordict[entry.name],edgecolor = edc)
    
                    if label_domains == True:
                        plt.annotate(entry.name, center, ha='center',
                                     va='center', **text_font,
                                     color = colordict[entry.name])
                
            plt.xlabel('Chem Pot '+projEle[0],**label_font)
            plt.ylabel('Chem Pot '+projEle[1],**label_font)
            for entry, vertices in eql._stable_domain_vertices.items():

                dataList = [vertices[:,i:i+1] for i in projIndex]
                data = np.hstack([dataList[0],dataList[1]])
#                 print(entry.name)
#                 print(data.tolist())
                hull = ConvexHull(data, qhull_options='QJ')
                points = [data[s] for s in hull.vertices]
                
                center = np.average(points, axis=0)
                points_centered = points - center
                # Sort points by cross product of centered points,
                # isn't strictly necessary but useful for plotting tools 
                point_comparator = lambda x, y: x[0] * y[1] - x[1] * y[0]
                points_centered = sorted(points_centered,
                                         key=cmp_to_key(point_comparator))
                points = points_centered + center
                x, y = np.transpose(np.vstack([points, points[0]]))
                if entry.name != "TaNO":
                    if entry.name == "Ta2O5":
                        plt.fill(x, y, alpha = 0.2,facecolor = colordict[entry.name],edgecolor = edc)
                    else:
                        plt.fill(x, y, alpha = alpha,facecolor = colordict[entry.name],edgecolor = edc)
                else: 
                    print(entry.name)
                    plt.fill(x, y, alpha = 0.3,facecolor = colordict[entry.name],edgecolor = edc,zorder=40)
                    plt.plot(x,y,c = colordict[entry.name], linewidth=2.5,
                             zorder=30,linestyle = "dashed")

                if label_equilLine:
                    plt.annotate(entry.name, center, ha='center',
                                 va='center', **text_font,
                                 color = colordict[entry.name],zorder=42)
#             plt.plot([-7,-0.7],[0.56,0.56],linestyle="dashed",c="red")
        if len(projEle) == 3:
            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            colordict = {}
            for e in self._cp._stable_domain_vertices:
                c = next(color)
                colordict[e.name] = list(c)
            
            '''set color based on Li atomic fraction'''
#             entries_names = []
#             for e in self._cp._stable_domain_vertices:
#                 stree = [str(ell) for ell in e.composition.elements]
#                 if "Li" in stree:
#                     entries_names.append(e.name)
#             entries_names.sort(key= lambda x: Composition(x).get_atomic_fraction("Li"), reverse=True)
#             jet= plt.get_cmap('plasma')
#             n = len(entries_names)
#             color=iter(jet(np.linspace(0,1,n)))
#             for e in entries_names:
#                 c = next(color)
#                 colordict[e] = list(c)
            
            if show_polytope:
                for e in self._cp._stable_domain_vertices:
#                     if e.name == "Ca":
                    data = self._cp._stable_domain_vertices[e]

                    dataList = [data[:,i:i+1] for i in projIndex]
                    data = np.hstack([dataList[0],dataList[1],dataList[2]])
                    hull = ConvexHull(data)
                    simplices = hull.simplices 
                    org_triangles = [data[s] for s in simplices]
                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = 0.1, facecolor=colordict[e.name],edgecolor = edc)
                    ax.add_collection3d(pc)
                    if label_domains == True:
                        
                        
                        vertices = [data[s] for s in hull.vertices]
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],e.name,ha='center',
                                va='center',**text_font,color = colordict[e.name])
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim([limits[0][0]-0.5,limits[0][1]+0.5])
            ax.set_ylim([limits[1][0]-0.5,limits[1][1]+0.5])
            ax.set_zlim([limits[2][0]-0.5,limits[2][1]+0.5])
            ax.set_xlabel('X: Chem Pot '+projEle[0],fontname='Arial',fontsize = 12)
            ax.set_ylabel('Y: Chem Pot '+projEle[1],fontname='Arial',fontsize = 12)
            ax.set_zlabel('Z: Chem Pot '+projEle[2],fontname='Arial',fontsize = 12)

            centers = {}
            fn = 0
            hull_3d = True
            for entry, vertices in eql._stable_domain_vertices.items():

                dataList = [vertices[:,i:i+1] for i in projIndex]
                data = np.hstack([dataList[0],dataList[1],dataList[2]])
                try:
                    '''if bold boundaries of 3d polytopes, hull can not be QJ option''' 
                    hull = ConvexHull(data)
                except:
                    '''always, 2d polygons (coplanar) can not be calculated in 3d.'''
                    hull_3d = False
#                         all_not_3d_names.append(names)
                    hull = ConvexHull(data,qhull_options = "QJ")
                simplices = hull.simplices 
                org_triangles = [data[s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, \
                     alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_equilLine:

                    cvertices = [data[s] for s in hull.vertices]
                    center = np.average(cvertices, axis=0)
                    centers[entry.name] = list(center)
                    ax.text(center[0],center[1],center[2],entry.name,ha='center',
                            va='center',**text_font,color = 'k')
                if bold_boundary:
                    if hull_3d:
                        lines1 = get_edges_of_polytope_in3D(org_triangles, hull)
                    else:
                        lines1 = get_edges_of_plane_in3D(vertices)
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.6,linewidth=1)
                fn += 1
#                 fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
#                                 "/myResearch/A220512_cp_goldsmith/figs/" +str(fn) )
        
        if title:
            projStr = ''
            for ee in projEle:
                projStr += ee + " "
            if compound != None:
                plt.title('Projection of '+compound + " system to "+ projStr,**label_font)
            else:
                eleStr = "( "
                for ii in self.elementList:
                    eleStr += ii +" "
                eleStr += ")" 
                plt.title('Projection of '+ eleStr + " system to "+ projStr,**label_font)
        plt.show()
        return plt
        
        
    def get_projection_plot(self,projEle, compound = None,
                            label_domains=True,edc = None, 
                            limits = None,alpha = 0.1):
        projIndex = [self.elementList.index(e) for e in projEle if e in self.elementList]
        F = len(projIndex)
        C = len(self.elementList)
        if limits == None:
            limits = []
            for i in range(F):
                limits += [[-10,0]]
        else:
            newlimits = []
            for stre in projEle:
                i = self.elementList.index(stre)
                newlimits += [limits[i]]
            limits = newlimits
            print(limits)
        label_font = {'fontname':'Calibri', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Calibri', 'size':'15', 'weight':'normal'}
        if len(projEle) == 2:

            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.set_xlim(limits[0])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                           colors='k',grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])
            jet= plt.get_cmap('gist_rainbow')
            n=len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            for entry, vertices1 in self._cp._stable_domain_vertices.items():
#                 if entry.name == "BaMnN2":
                dataList = [vertices1[:,i:i+1] for i in projIndex]
                data = np.hstack([dataList[0],dataList[1]])
                hull = ConvexHull(data)
                vertices = [data[s] for s in hull.vertices]
#                 print(entry.name,vertices)
                center = np.average(vertices, axis=0)
                x, y = np.transpose(np.vstack([vertices, vertices[0]]))
#                 print(np.vstack([vertices, vertices[0]]))
#                 print(np.transpose(np.vstack([vertices, vertices[0]])))
                c=next(color)
                plt.fill(x, y, alpha = 0.1,facecolor = c,edgecolor = edc)
            #             label_domains=False
                if label_domains == True:
                    plt.annotate(entry.name, center, ha='center',
                                     va='center', **text_font,color = c)
                
            plt.xlabel('Chem Pot '+projEle[0],**label_font)
            plt.ylabel('Chem Pot '+projEle[1],**label_font)
                
        if len(projEle) == 3:
            jet= plt.get_cmap('gist_rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            for e in self._cp._stable_domain_vertices:
                c=next(color)
            #     if e.name ==compound:
                data = self._cp._stable_domain_vertices[e]
#                 print('data',data)
                dataList = [data[:,i:i+1] for i in projIndex]
                data = np.hstack([dataList[0],dataList[1],dataList[2]])
                hull = ConvexHull(data)
                simplices = hull.simplices 
                org_triangles = [data[s] for s in simplices]
             
                pc = a3.art3d.Poly3DCollection(org_triangles, alpha = alpha, facecolor=c,edgecolor = edc)
                ax.add_collection3d(pc)
                vertices = [data[s] for s in hull.vertices]
                center = np.average(vertices, axis=0)
                print(vertices)
                text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
            #     if e.name == compound or e.name == "Ba5B3ClO9":
                if label_domains == True:
                    ax.text(center[0],center[1],center[2],e.name,ha='center',
                            va='center',**text_font,color = c)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            ax.set_xlabel('X: Chem Pot '+projEle[0],**text_font)
            ax.set_ylabel('Y: Chem Pot '+projEle[1],**text_font)
            ax.set_zlabel('Z: Chem Pot '+projEle[2],**text_font)
        projStr = ''
        for ee in projEle:
            projStr += ee + " "
        if compound != None:
            plt.title('Projection of '+compound + " system to "+ projStr, **text_font)
        else:
            eleStr = "( "
            for ii in self.elementList:
                eleStr += ii +" "
            eleStr += ")" 
            plt.title('Projection of '+ eleStr + " system to "+ projStr,**text_font)
        plt.show()
    
    
    def get_slice_equilLine_plot(self,limits=None, compound = None,
                     title="",label_domains=False,edc = None, alpha = 0.5,
                     show_polytope = True,label_equilLine = True):
        eql = EquilLine(self._cp._processed_entries,self.elementList,
                        fixed = self.fixed, limits = limits)
        ecopy = self.elementList.copy()
        for e in self.fixed:
            ecopy.remove(e[0])
        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]
        if limits == None:
            limits = []
            for i in range(C-F):
                limits += [[-10,0]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex:
                    newlimits += [limits[i]]
            limits = newlimits
            print(limits)
        label_font = {'fontname':'Calibri', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Calibri', 'size':'15', 'weight':'normal'}
        if len(ecopy) == 2:
            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.set_xlim(limits[0])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])

            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            colorr=iter(jet(np.linspace(0,1,n)))
            colordict = {}
            for e in self._cp._stable_domain_vertices:
                c = next(colorr)
                colordict[e.name] = c
            if show_polytope:
                for entry, vertices in self._cp._stable_domain_vertices.items():
    #                 if entry.name == 'BaN2':
                    center = np.average(vertices, axis=0)
                    x, y = np.transpose(np.vstack([vertices, vertices[0]]))
                    plt.fill(x, y, alpha = 0.1,facecolor = colordict[entry.name])
#                     if label_domains:
                    plt.annotate(generate_entry_label(entry), center, ha='center',
                                 va='center', **text_font,color = colordict[entry.name])        
            plt.xlabel('Chem Pot '+ecopy[0],**label_font)
            plt.ylabel('Chem Pot '+ecopy[1],**label_font)
            for entry, vertices in eql._stable_domain_vertices.items():
                plt.plot(vertices[:,:1],vertices[:,1:2],linewidth = 2.25,color = colordict[entry.name])
#             plt.show()
        if len(ecopy) == 3:
            
            jet= plt.get_cmap('gist_rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            ax.grid(b=None)
            CP_domain_vertices = self._cp._stable_domain_vertices
            colordict = {}
            for e in CP_domain_vertices:
                c = next(color)
                colordict[e.name] = c.tolist()
            print("colordict",colordict)
            if show_polytope:
                for e in CP_domain_vertices:
    #                 if e.name == 'BaMnN2' or e.name == "Ba3MnN3":
                    hull = ConvexHull(CP_domain_vertices[e])
                    simplices = hull.simplices
                    org_triangles = [CP_domain_vertices[e][s] for s in simplices]
                    pc = a3.art3d.Poly3DCollection(org_triangles, alpha = 0.1, 
                                                   facecolor=colordict[e.name],edgecolor = edc)
                    ax.add_collection3d(pc)
                    if label_domains:
                        center = np.average(CP_domain_vertices[e], axis=0)
                        ax.text(center[0],center[1],center[2],e.name,ha='center',
                                va='center',**text_font,color = colordict[e.name])
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
#             ax.set_xlabel('Chem Pot '+ecopy[0],fontname='Arial',fontsize = 15)
#             ax.set_ylabel('Chem Pot '+ecopy[1],fontname='Arial',fontsize = 15)
#             ax.set_zlabel('Chem Pot '+ecopy[2],fontname='Arial',fontsize = 15)
            hull = ConvexHull(eql.intersection)
            simplices = hull.simplices
            org_triangles = [eql.intersection[s] for s in simplices]
            triDict = {entry: [] for entry in eql._processed_entries}
            for entry, vertices in eql._stable_domain_vertices.items():
                for tri in org_triangles:
                    if all(x in vertices.tolist() for x in tri.tolist()):
                        triDict[entry].append(tri)
                # print(triDict[entry])
                
            for entry in eql._processed_entries:
                if entry.name=="B":
                    triDict[entry]=[np.array([[0, -10.0, -10], [0.0, -1.952364127500005, -10],[0, -10.0, -5.636889556249995]]),
                                    np.array([[0.0, -1.952364127500005, -10],[0, -10.0, -5.636889556249995], [0.0, -1.9523641275000005, -5.636889556249997]])
                                   ]
                    pc = a3.art3d.Poly3DCollection(triDict[entry], alpha = alpha,
                                                   facecolor = colordict[entry.name],edgecolor = edc)
                    ax.add_collection3d(pc)
            triDict = {k: v for k, v in triDict.items() if v}
            for entry, vertices in eql._stable_domain_vertices.items():
#                 if entry.name == "Fe":
                pc = a3.art3d.Poly3DCollection(triDict[entry], alpha = alpha,
                                               facecolor = colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_equilLine:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],entry.name,c='k',ha='center',
                            va='center',**text_font,color = "k")#colordict[entry.name])
            org_triangles = [eql.intersection[s] for s in simplices]
            print(org_triangles)
#             print(len(org_triangles))

            '''print edge'''
            linesDict = {entry: [] for entry in triDict}
            for entry in triDict:

                lines = []
                for tri in triDict[entry]:
                    lines += list(combinations(tri.tolist(),2))
                lines1 = []
                for i in lines:
                    if i not in lines1:
                        lines1.append(i)
                    else:
                        lines1.remove(i)
                linesDict[entry] = lines1
                print()
                print(len(lines1))
                for ii in lines1:
                    print(ii)
                # if entry.name=="BaCl2":
                #     GAGA
                for li in lines1:
                    li = np.array(li)
                    ax.plot(li[:,0],li[:,1],li[:,2],color="k",alpha = 0.7,linewidth=2.5,
                            )
                
        
        
        if self.sliceTitle == True:
            fixedStr = ''
            numStr = ''
            for e in self._cp.fixedEle:
                fixedStr += e[0] + " "
                numStr += str(e[1])+','
            if compound != None:
                title1 = "Slice of "+compound+" system with "+fixedStr+"mu fixed at " +numStr
            else:
                eleStr = "( "
                for ii in self._cp.elementList:
                    eleStr += ii +" "
                eleStr += ")" 
                title1 = "Slice of "+eleStr+" system with "+fixedStr+"mu fixed at " +numStr
            if title == "":
                title = title1
        plt.title(title, fontname = 'Arial', fontsize=15)
        plt.show()
    
    def get_scaled_equil_line_on_CP_Halfspace(self, ww=None,limits=None, compound = None,
                         title="",label_domains=False,edc = None, 
                         alpha = 0.2, show_polytope = True,
                         label_equilLine=True,show_label=True):

        eql = EquilLine(self._cp._processed_entries,self.elementList,fixed = None, limits = limits)
        if len(self.elementList)==3:
            hull = ConvexHull(eql.intersection)
            simplices = hull.simplices
            org_triangles1 = [eql.intersection[s] for s in simplices]
            triDict1 = {entry: [] for entry in eql._processed_entries}
            for entry, vertices in eql._stable_domain_vertices.items():
                for tri in org_triangles1:
                    if all(x in vertices.tolist() for x in tri.tolist()):
                        triDict1[entry].append(tri)
            triDict1 = {k: v for k, v in triDict1.items() if v}
        for e in eql._stable_domain_vertices:
            X_scaled = eql._stable_domain_vertices[e]
            print(e.name, X_scaled)
            vec_list = []
            for i in range(len(X_scaled)):
                vec_new = (np.array(X_scaled[i])).dot(np.linalg.inv(np.array(ww)))
                vec_list.append(vec_new)
            print(e.name, np.array(vec_list))
            eql._stable_domain_vertices[e] = np.array(vec_list)

        label_font = {'fontname':'Calibri', 'size':'20', 'color':'black', 'weight':'normal'}
    #               'verticalalignment':'bottom'}
        text_font = {'fontname':'Calibri', 'size':'20', 'weight':'normal'}
        if len(self.elementList) == 2:
            if limits is None:
                limits = [[-10, 0], [-10, 0]]
            
            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
#             ax.set_xlim(limits[0])

            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
#             ax.set_ylim(limits[1])
            colordict = {}
            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            colorr=iter(jet(np.linspace(0,1,n)))
            cpentries = list(self._cp._stable_domain_vertices.keys())
            cpentries = sorted(cpentries,key = lambda e:e.entry.composition.get_atomic_fraction(self.elementList[0]))
            for e in cpentries:
                c = next(colorr)
                colordict[e.name] = c
                print(e.name, c)
            if show_polytope:
                for entry in cpentries:
                    vertices = self._cp._stable_domain_vertices[entry]
    #                 if entry.name == 'BaN2':
                    center = np.average(vertices, axis=0)
                    print(vertices)
                    print(entry.name,center)
                    x, y = np.transpose(np.vstack([vertices, vertices[0]]))
                    plt.fill(x, y, alpha = alpha,facecolor = colordict[entry.name])
#                     if label_domains:
#                     plt.annotate(generate_entry_label(entry), center, ha='center',
#                                  va='center', **text_font,color = colordict[entry.name])
            
            if show_label:
                plt.xlabel('Chem Pot '+self.elementList[0]+"-"+self.elementList[-1],**label_font)
                plt.ylabel('Chem Pot -'+self.elementList[1],**label_font)
#             else:
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
            for entry, vertices in eql._stable_domain_vertices.items():
                plt.plot(vertices[:,:1],vertices[:,1:2],linewidth = 2.25,color = colordict[entry.name])
            plt.show()
        if len(self.elementList) == 3:
            jet= plt.get_cmap('gist_rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            if limits is None:
                limits = [[-10, 0], [-10, 0], [-10, 0]]
            CP_domain_vertices = self._cp._stable_domain_vertices
            colordict = {}
            cpentries = list(self._cp._stable_domain_vertices.keys())
            cpentries = sorted(cpentries,key = lambda e:e.entry.composition.get_atomic_fraction(self.elementList[0]))
            for e in cpentries:
                c = next(color)
                colordict[e.name] = c


            if show_polytope:
                for e in cpentries:
#                     if e.name == 'Li4GeS4':
                    hull = ConvexHull(CP_domain_vertices[e])
                    simplices = hull.simplices
                    org_triangles = [CP_domain_vertices[e][s] for s in simplices]
                    pc = a3.art3d.Poly3DCollection(org_triangles, alpha = 0.1, 
                                                   facecolor=colordict[e.name],edgecolor = edc)
                    ax.add_collection3d(pc)
                    if label_domains:
                        center = np.average(CP_domain_vertices[e], axis=0)
                        ax.text(center[0],center[1],center[2],e.name,ha='center',
                                va='center',**text_font)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim([0,-2*limits[-1][0]])
#             ax.set_xticks([i for i in range(limits[0][0], limits[0][1])])
#             ax.set_yticks([i for i in range(limits[1][0], limits[1][1])])
#             ax.set_zticks([i for i in range(0,-2*limits[-1][0])])
            if show_label:
                ax.set_xlabel('Chem Pot '+self.elementList[0]+"-"+self.elementList[-1],fontname='Arial',fontsize = 15)
                ax.set_ylabel('Chem Pot '+self.elementList[1]+"-"+self.elementList[-1],fontname='Arial',fontsize = 15)
                ax.set_zlabel('Chem Pot -'+self.elementList[2],fontname='Arial',fontsize = 15)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            plt.title(title, fontname = 'Arial', fontsize=15,verticalalignment='center')

            triDict = {entry: [] for entry in eql._processed_entries}
            for entry, vertices in eql._stable_domain_vertices.items():
                hull=ConvexHull(vertices,qhull_options='QJ')
                simplices = hull.simplices
                triDict[entry] = [vertices[s] for s in simplices]
            triDict = {k: v for k, v in triDict.items() if v}
            
            '''print edge'''

            linesDict = {entry: [] for entry in triDict1}
            for entry in triDict1:
                print(entry.name)
                lines = []
                for tri in triDict1[entry]:
                    lines += list(combinations(tri.tolist(),2))
                lines1 = []
                for i in lines:
                    if i not in lines1:
                        lines1.append(i)
                    else:
                        lines1.remove(i)
                linesDict[entry] = lines1

                print(len(lines1))
                for ii in lines1:
                    print(ii)
                for li in lines1:
                    li = np.array([(np.array(jjj)).dot(np.linalg.inv(np.array(ww)))for jjj in li])
                    ax.plot(li[:,0],li[:,1],li[:,2],color="k",alpha = 0.5,linewidth=2)
            
            '''plot eql face'''
            for entry, vertices in eql._stable_domain_vertices.items():
                print(entry.name)
                pc = a3.art3d.Poly3DCollection(triDict[entry], alpha = alpha,
                                               facecolor = colordict[entry.name])#,edgecolor = 'w')
                ax.add_collection3d(pc)
                if label_equilLine:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],entry.name,c='k',ha='center',
                            va='center',**text_font)
            plt.show() 
       
    def get_mu_xxx_plot2(self,projEle, 
                        show_range = None,
                            label_single=False, 
                            limits = None,alpha = 0.2,
                            axis_off = 0,
                            label_phase2=False, label_phase3=False):
        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
    #               'verticalalignment':'bottom'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        
        els = set(self.elementList).difference(projEle)
        elements = [Element(el) for el in list(els)]
        all_entries = [e for e in self._cp._processed_entries if len(els.intersection([str(el) for el in e.composition.elements])) > 0]

        eql = EquilLine(all_entries,self.elementList,fixed = None, limits = limits)
        projIndex = [self.elementList.index(e) for e in projEle if e in self.elementList][0]

        stable_domain_vertices = eql._stable_domain_vertices
        fig = plt.figure(figsize=(9.2, 7))
        ax = a3.Axes3D(fig)
        
        murange = {}
        for entry, vertices in stable_domain_vertices.items():
            print(entry.name)
            maxv = np.max(vertices[:, projIndex])
            minv = np.min(vertices[:, projIndex])
            murange[entry] = (max(minv,show_range[0]), 
                              min(maxv,show_range[1]))
        segments = overlap_lines_axis(murange)
        print(len(segments))
        
        coord = {}
#         for entry in murange:
        for s,es in segments.items():

            if s[1] > show_range[1]: 
                # we only visualize partial muxxx diagram when the block minimum muN 
                # is greater or equal than show range minimum 
                continue
            if s[0] < show_range[0]:
                continue
#                 s = list(s)
#                 s[0] = show_range[0]
            '''This grand pot diagram is based on formation energy, not dft energy like pymatgen does'''
            gpd = GraPotPhaseDiagram(es, {self.elementList[projIndex]:(s[0] + s[1])/2})
            plotter = PDPlotter(gpd)
            lines, stable, unstable = plotter.pd_plot_data
            for x,y in stable:
                entry = stable[(x, y)]
                if entry not in coord:
                     
                    coord[entry.name] = [x,y]

        muname_range = {entry.name:murange[entry] for entry in murange}
        '''get triangle color dict'''
        colordict = {}
        symbols = {e.name : e.symbol for e in murange}
        for name in coord:

            zmin = max(muname_range[name][0], show_range[0])
#             zmin = max(muname_range[name][0], -1.2)
            zmax = min(muname_range[name][1], show_range[1])
            color = get_color_tri(coord[name]).tolist() + [(abs(zmin) + 0.5) / 6]
            color = [round(c,4) for c in color]
            colordict[name] = np.array(color)
#             if name not in ["Li2CoNi3O8", "Li(NiO2)2", "Ni3O4","LiO8"]:
#                 continue
            x = coord[name][0]
            y = coord[name][1]
            ax.plot([x, x], [y, y], 
                    [zmin, zmax], 
                    c = colordict[name],linewidth=5)
            '''add label'''
            if label_single:
                
                ax.text(x, y, (zmin + zmax) / 2, symbols[name],
                        c='k',ha='center',
                        va='center',**text_font)

        print(colordict)
        print(len(colordict))

        if show_range[0] < limits[projIndex][0]:
            ax.set_zlim(limits[projIndex])
        else:
            ax.set_zlim(show_range)
        ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                       colors='k',grid_color='k', grid_alpha=0.5)
        ax.grid(b=None)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.set_zlabel("mu" + projEle[0], **label_font)
        exy_loc = {}
        for s,es in segments.items():

            if s[1] > show_range[1]: 
                # we only visualize partial muxxx diagram when the block minimum muN 
                # is greater or equal than show range minimum 
                continue
            if s[0] < show_range[0]:
                continue
#                 s = list(s)
#                 s[0] = show_range[0]
            '''This grand pot diagram is based on formation energy, not dft energy like pymatgen does'''
            print(s, [i.name for i in es])
            gpd = GraPotPhaseDiagram(es, {self.elementList[projIndex]:(s[0] + s[1])/2})
            plotter = PDPlotter(gpd)
            lines, stable, unstable = plotter.pd_plot_data
            
            '''2-phase, vertical plane'''    
            for x, y in lines:
#                 ax.plot(x,y,s[1],"k-")
                   
                tris = [
                    [[x[0], y[0], s[0]],
                     [x[1], y[1], s[0]],
                     [x[0], y[0], s[1]]],
                    [[x[0], y[0], s[1]],
                     [x[1], y[1], s[0]],
                     [x[1], y[1], s[1]]]
                ]
                   
                v1 = (x[0], y[0])
                v2 = (x[1], y[1])
                  
                entry1 = stable[v1]
                entry2 = stable[v2]
#                 if entry1.name not in ["Li2CoNi3O8", "Li(NiO2)2", "Ni3O4","LiO8"] or \
#                     entry2.name not in ["Li2CoNi3O8", "Li(NiO2)2", "Ni3O4","LiO8"]:
#                     continue
#                 print(colordict[entry1.name],colordict[entry2.name])
                fc = (colordict[entry1.name] + colordict[entry2.name])/2
                pc = a3.art3d.Poly3DCollection(tris, alpha = alpha, 
                                                facecolor = fc)
                ax.add_collection3d(pc)
        if axis_off:
            ax.set_axis_off()
        plt.show()
        
    def get_mu_xxx_plot(self,projEle, 
                        show_range = None,
                            label_single=False, 
                            limits = None,alpha = 0.2,
                            label_phase2=False, label_phase3=False):
        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
    #               'verticalalignment':'bottom'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        
        els = set(self.elementList).difference(projEle)
        elements = [Element(el) for el in list(els)]
        all_entries = [e for e in self._cp._processed_entries if len(els.intersection([str(el) for el in e.composition.elements])) > 0]

        eql = EquilLine(all_entries,self.elementList,fixed = None, limits = limits)
        projIndex = [self.elementList.index(e) for e in projEle if e in self.elementList][0]

        stable_domain_vertices = eql._stable_domain_vertices
        murange = {}
        for entry, vertices in stable_domain_vertices.items():
            print(entry.name)
            maxv = np.max(vertices[:, projIndex])
            minv = np.min(vertices[:, projIndex])
            murange[entry] = (minv, maxv)
        segments = overlap_lines_axis(murange)
        print(len(segments))
#         for s, es in segments.items():
#             print(s)
#             print([e.name for e in es])
#             print()
        colordict = get_colors(stable_domain_vertices.keys(), cmap = 'rainbow')

        fig = plt.figure(figsize=(9.2, 7))
        ax = a3.Axes3D(fig)
        if show_range[0] < limits[projIndex][0]:
            ax.set_zlim(limits[projIndex])
        else:
            ax.set_zlim(show_range)
        ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                       colors='k',grid_color='k', grid_alpha=0.5)
        ax.grid(b=None)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.set_zlabel("mu" + projEle[0], **label_font)
        exy_loc = {}
        for s,es in segments.items():

            if s[1] > show_range[1]: 
                # we only visualize partial muxxx diagram when the block minimum muN 
                # is greater or equal than show range minimum 
                continue
            if s[0] < show_range[0]:
                continue
#                 s = list(s)
#                 s[0] = show_range[0]
            '''This grand pot diagram is based on formation energy, not dft energy like pymatgen does'''
            print(s, [i.name for i in es])
            gpd = GraPotPhaseDiagram(es, {self.elementList[projIndex]:(s[0] + s[1])/2})
            plotter = PDPlotter(gpd)
            lines, stable, unstable = plotter.pd_plot_data
            
            '''2-phase, vertical plane'''    
            for x, y in lines:
#                 ax.plot(x,y,s[1],"k-")
                  
                tris = [
                    [[x[0], y[0], s[0]],
                     [x[1], y[1], s[0]],
                     [x[0], y[0], s[1]]],
                    [[x[0], y[0], s[1]],
                     [x[1], y[1], s[0]],
                     [x[1], y[1], s[1]]]
                ]
                  
                v1 = (x[0], y[0])
                v2 = (x[1], y[1])
                 
                entry1 = stable[v1]
                entry2 = stable[v2]
                fc = (colordict[entry1.name] + colordict[entry2.name])/2
                pc = a3.art3d.Poly3DCollection(tris, alpha = alpha, 
                                                facecolor = fc)
                ax.add_collection3d(pc)
            
            '''single phase, vertical line'''
            for x,y in stable:
                entry = stable[(x, y)]
#                 print(entry.name)
#                 print(x, y)
                
                ax.plot([x, x], [y, y], s, c = colordict[entry.name],
                        linewidth=5)
                if entry.name not in exy_loc:
                    exy_loc[entry.name] = (x, y)
        '''add label'''
        if label_single:
            mr = {e.name : murange[e] for e in murange}
            symbols = {e.name : e.symbol for e in murange}
            for name in exy_loc:
                x, y = exy_loc[name]
                z = (max(show_range[0], mr[name][0]) + min(mr[name][1], show_range[1])) / 2
                ax.text(x, y, z, symbols[name],
                        c='k',ha='center',
                        va='center',**text_font)

        plt.show()
        
    def get_equil_line_on_CP_Halfspace(self, limits=None, compound = None,
                         title="", edc = None, 
                         alpha = 0.2, show_polytope = True,
                         label_domain=True, show_label=True,
                         label_equilLine = True):

        eql = EquilLine(self._cp._processed_entries,self.elementList,fixed = None, limits = limits)
        label_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal'}
    #               'verticalalignment':'bottom'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(self.elementList) == 2:
            if limits is None:
                limits = [[-10, 0], [-10, 0]]
            
            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.set_xlim(limits[0])
#             ax.set_xticks([-8,-6,-4,-2,0])
#             ax.set_yticks([-8,-6,-4,-2,0])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])
            colordict = {}
            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            colorr=iter(jet(np.linspace(0,1,n)))
            cpentries = list(self._cp._stable_domain_vertices.keys())
            cpentries = sorted(cpentries,key = lambda e:e.entry.composition.get_atomic_fraction(self.elementList[0]))
            colors = []
            for e in cpentries:
                c = next(colorr)
                colordict[e.name] = c.tolist()
                colors.append(c.tolist())
                print(e.name, c)

            print("colordict")
            print(colordict)

            if show_polytope:
                for entry in cpentries:
                    vertices = self._cp._stable_domain_vertices[entry]
    #                 if entry.name == 'BaN2':
                    center = np.average(vertices, axis=0)
                    print(vertices)
                    print(entry.name,center)
                    x, y = np.transpose(np.vstack([vertices, vertices[0]]))
                    plt.fill(x, y, alpha = alpha,facecolor = colordict[entry.name])
                    
                    if label_domain:
                        plt.annotate(entry.symbol, center, ha='center',
                                     va='center', **text_font,color = colordict[entry.name],
                                     zorder=40)
#             for axis in ['top','bottom','left','right']:
#                 ax.spines[axis].set_linewidth(2)

            if show_label:
                plt.xlabel('Chem Pot '+self.elementList[0],**label_font)
                plt.ylabel('Chem Pot '+self.elementList[1],**label_font)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            for entry, vertices in eql._stable_domain_vertices.items():
                ww = 3
                plt.plot(vertices[:,:1],vertices[:,1:2],linewidth = ww,color = colordict[entry.name])
                
            plt.show()
        if len(self.elementList) == 3:
            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            if limits is None:
                limits = [[-10, 0], [-10, 0], [-10, 0]]
            CP_domain_vertices = self._cp._stable_domain_vertices
            colordict = {}
            cpentries = list(self._cp._stable_domain_vertices.keys())
            cpentries = sorted(cpentries,key = lambda e:e.entry.composition.get_atomic_fraction(self.elementList[0]))
            for e in cpentries:
                c = next(color)
                colordict[e.name] = c.tolist()
            print(colordict)
            colordict = {'MnO': [0.5, 0.0, 1.0, 1.0], 'MnO2': [0.37450980392156863, 0.19584546700716696, 0.9951469164070644, 1.0], 'Li9Mn20O40': [0.24901960784313726, 0.38410574917192586, 0.9806347704689777, 1.0], 'Li2MnO3': [0.12352941176470589, 0.5574894393428855, 0.9566044195004408, 1.0], 'Mn3O4': [0.0019607843137254832, 0.7092813076058535, 0.9232891061054894, 1.0], 'Li6MnO4': [0.12745098039215685, 0.8336023852211195, 0.8810121942857845, 1.0], 'O2': [0.2529411764705882, 0.9256376597815562, 0.8301840308155507, 1.0], 'Li2O': [0.3784313725490196, 0.9818225628535369, 0.7712979623471807, 1.0], 'Mn2O3': [0.503921568627451, 0.9999810273487268, 0.7049255469061472, 1.0], 'Li4Mn5O12': [0.6294117647058823, 0.9794097676013659, 0.631711006253251, 1.0], 'Mn': [0.7549019607843137, 0.9209055179449537, 0.5523649729605058, 1.0], 'LiMn2O4': [0.8803921568627451, 0.8267341748257635, 0.4676575928925868, 1.0], 'Li2O2': [1.0, 0.7005430375932911, 0.37841105004231035, 1.0], 'LiO8': [1.0, 0.5472195469221114, 0.28549158627534216, 1.0], 'Li9Mn14O32': [1.0, 0.37270199199091436, 0.18980109344182594, 1.0], 'Li11Mn13O32': [1.0, 0.18374951781657037, 0.09226835946330202, 1.0], 'LiMnO2': [1.0, 1.2246467991473532e-16, 6.123233995736766e-17, 1.0]}            
            namess = ["LiMn2O4","LiMnO2"]
            if show_polytope:
                for e in cpentries:
                    if e.name not in namess:
                        continue
                    eql_vertices = eql._stable_domain_vertices[e]
                    
                    plane_normal = calculate_plane_normal(eql_vertices[0],eql_vertices[1],eql_vertices[2])
                    on_plane, on_positive_side, on_negative_side = separate_points_by_plane(CP_domain_vertices[e], 
                                                                                            plane_normal, 
                                                                                            eql_vertices[0])
#                     print(on_positive_side)
#                     print(on_positive_side + eql_vertices)
                    for side, alpha1 in zip([on_positive_side, on_negative_side], [0.08,0.4]):
                        half_vertices = np.concatenate((eql_vertices, side), axis=0)
                        hull = ConvexHull(half_vertices)
                        
                        simplices = hull.simplices
                        org_triangles = [half_vertices[s] for s in simplices]
                        for orgg in org_triangles:
                            print(orgg)
                        if e.name == "LiMnO2" and alpha1 == 0.4:
                            alpha1 = 0.25
                        if e.name == "LiMnO2" and alpha1 == 0.08:
                            alpha1 = 0.05
                        pc = a3.art3d.Poly3DCollection(org_triangles, alpha = alpha1, 
                                                       facecolor=colordict[e.name],edgecolor = edc)
                        
                        ax.add_collection3d(pc)
                        
                        center = np.average(half_vertices, axis=0)
                        # ax.text(center[0],center[1],center[2],e.name,ha='center',
                        #         va='center',zorder=40,**text_font)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            if show_label:
                print()
#                 ax.set_xlabel('Chem Pot '+self.elementList[0],fontname='Arial',fontsize = 15)
#                 ax.set_ylabel('Chem Pot '+self.elementList[1],fontname='Arial',fontsize = 15)
#                 ax.set_zlabel('Chem Pot '+self.elementList[2],fontname='Arial',fontsize = 15)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xticks(np.arange(limits[0][0],limits[0][1]+0.01,step = 0.5))
                ax.set_yticks(np.arange(limits[1][0],limits[1][1]+0.01,step = 0.5))
                ax.set_zticks(np.arange(limits[2][0],limits[2][1]+0.01,step = 0.5))
            plt.title(title, fontname = 'Arial', fontsize=15,verticalalignment='center')
            hull = ConvexHull(eql.intersection)
            simplices = hull.simplices

            org_triangles = [eql.intersection[s] for s in simplices]
            print(org_triangles)
#             print(len(org_triangles))
            triDict = {entry: [] for entry in eql._processed_entries}
            for entry, vertices in eql._stable_domain_vertices.items():
                for tri in org_triangles:
                    if all(x in vertices.tolist() for x in tri.tolist()):
                        triDict[entry].append(tri)
            triDict = {k: v for k, v in triDict.items() if v}

            '''print edge'''
            linesDict = {entry: [] for entry in triDict}
            for entry in triDict:

                lines = []
                for tri in triDict[entry]:
                    lines += list(combinations(tri.tolist(),2))
                lines1 = []
                for i in lines:
                    if i not in lines1:
                        lines1.append(i)
                    else:
                        lines1.remove(i)
                linesDict[entry] = lines1

                for li in lines1:
                    li = np.array(li)
                    ax.plot(li[:,0],li[:,1],li[:,2],color="k",alpha = 0.7,linewidth=2.5,
                            )

            '''plot eql face'''
            for entry, vertices in eql._stable_domain_vertices.items():

                pc = a3.art3d.Poly3DCollection(triDict[entry], alpha = alpha,
                                               facecolor = colordict[entry.name])#,edgecolor = 'w')
                ax.add_collection3d(pc)
                if label_equilLine:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],entry.symbol,c='k',ha='center',
                            va='center',**text_font)
            ax.text(-6,-6,-5,"Dissolution",c='gray',ha='center',
                va='center',**label_font)
            ax.text(-5,0,0,"Growth",c='k',ha='center',
                va='center',**label_font)
            plt.show() 
#             return plt
    
    
    
    def get_equil_line_on_CP_Intercept(self,limits=None, compound = None,
                         title="",label_domains=True,edc = None, 
                         alpha = 0.1,filename = None,PDentries = None):
        if len(self.elementList) == 2:
            print(self.elementList)
            if limits is None:
                limits = [[-10, 0], [-10, 0]]
            
            fig = plt.figure(figsize=(9.2, 7))
            label_font = {'fontname':'Calibri', 'size':'20', 'color':'black', 'weight':'normal'}
    #               'verticalalignment':'bottom'}
            text_font = {'fontname':'Calibri', 'size':'15', 'weight':'normal'}
            ax = plt.gca()
            xlim = limits[0]
            ylim = limits[1]
            ax.set_xlim(limits[0])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])

            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            colorr=iter(jet(np.linspace(0,1,n)))
            colorlist = []
            for entry, vertices in self._cp._stable_domain_vertices.items():
#                 if entry.name == 'BaN2':
                center = np.average(vertices, axis=0)
                x, y = np.transpose(np.vstack([vertices, vertices[0]]))
                c=next(colorr)
                plt.fill(x, y, alpha = 0.2,facecolor = c)
                colorlist.append(c)
    #             label_domains=False
                if label_domains:
                    plt.annotate(generate_entry_label(entry), center, ha='center',
                                 va='center', **text_font,color = c)
            els = self.elementList 
            CPentries = self._cp._processed_entries
            xx = [xlim[1]]
            yy = [ylim[0]]      
            for e in CPentries:
                formE = e.form_E
                ind = CPentries.index(e)
                if ind != len(CPentries)-1:
                    formEnext = CPentries[ind+1].form_E
                    slope = (formEnext-formE)/(CPentries[ind+1].entry.composition.get_atomic_fraction(els[1])-e.entry.composition.get_atomic_fraction(els[1]))
                    intercept = formE-slope*e.entry.composition.get_atomic_fraction(els[1])
                    xx.append(intercept)
                    yy.append(intercept+slope)
            xx.append(xlim[0])
            yy.append(ylim[1])
            
            xy = np.column_stack((xx,yy))
            print(xy)
            index = 0
            for start, stop in zip(xy[:-1], xy[1:]):
                xxx, yyy = zip(start, stop)
                ax.plot(xxx,yyy,linewidth = 2.25,color = colorlist[index])
                print(xxx,yyy)
                index += 1
            plt.xlabel('Chem Pot '+self.elementList[0],**label_font)
            plt.ylabel('Chem Pot '+self.elementList[1],**label_font)
            if filename != None:
                fig.savefig(filename+"_chem_pot_equi_line",dpi = 200)
            plt.show()
        if len(self.elementList) == 3:
            if PDentries == None:
                raise ValueError("You should put in phase diagram entries")
            jet= plt.get_cmap('gist_rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            if limits is None:
                limits = [[-10, 0], [-10, 0], [-10, 0]]
            CP_domain_vertices = self._cp._stable_domain_vertices
#             showList = ["N2","MnN","BaN6"]
            for e in CP_domain_vertices:
#                 if e.name in showList:
                hull = ConvexHull(CP_domain_vertices[e])
                simplices = hull.simplices
                org_triangles = [CP_domain_vertices[e][s] for s in simplices]
                c=next(color)
                pc = a3.art3d.Poly3DCollection(org_triangles, alpha = alpha, facecolor=c,edgecolor = edc)
                ax.add_collection3d(pc)
                center = np.average(CP_domain_vertices[e], axis=0)

                print("center",center)
                text_font = {'fontname':'Consolas', 'size':'15', 'color':c, 'weight':'normal'}
                ax.text(center[0],center[1],center[2],e.name,ha='center',
                        va='center',**text_font)
            elsE = [Element(i) for i in self.elementList]
            cppdata = PDPlotter(PhaseDiagram(PDentries,elsE)).tern_facets_chem_pot_data(limits = limits)
            
            hull = ConvexHull(cppdata)
            simplices = hull.simplices
            org_triangles = [cppdata[s] for s in simplices]
            pc = a3.art3d.Poly3DCollection(org_triangles, alpha = 0.3, facecolor="k", edgecolor = 'w')
            ax.add_collection3d(pc)
            for cpdata in cppdata:
                ax.scatter(cpdata[0],cpdata[1],cpdata[2],c = 'k',s = 30)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            ax.set_xlabel('Chem Pot '+self.elementList[0],fontname='Consolas',fontsize = 12)
            ax.set_ylabel('Chem Pot '+self.elementList[1],fontname='Consolas',fontsize = 12)
            ax.set_zlabel('Chem Pot '+self.elementList[2],fontname='Consolas',fontsize = 12)
            plt.show()
            
    def get_slice_plot(self,limits=None, compound = None,
                         title="",label_domains=True,edc = None, alpha = 0.1):
        ecopy = self.elementList.copy()
        for e in self.fixed:
            ecopy.remove(e[0])
        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]
        if limits == None:
            limits = []
            for i in range(C-F):
                limits += [[-10,0]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex:
                    newlimits += [limits[i]]
            limits = newlimits
            print(limits)
        label_font = {'fontname':'Arial', 'size':'20', 'color':'black'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(ecopy) == 2:
            plt.figure(figsize=(9.2, 7))
            xlim = limits[0]
            ylim = limits[1]
            ax = plt.gca()
            ax.set_xlim(xlim)
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            ax.set_ylim(ylim)

            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            colorr=iter(jet(np.linspace(0,1,n)))
            for entry, vertices in self._cp._stable_domain_vertices.items():
#                 if entry.name == 'BaN2':
                center = np.average(vertices, axis=0)
                c=next(colorr)
                x, y = np.transpose(np.vstack([vertices, vertices[0]]))
                plt.fill(x, y, alpha = 0.2,facecolor = c)
                if label_domains:
                    plt.annotate(generate_entry_label(entry), center, ha='center',
                                 va='center', **text_font,color = c)        
            plt.xlabel('Chem Pot '+ecopy[0],**label_font)
            plt.ylabel('Chem Pot '+ecopy[1],**label_font)
        if len(ecopy) == 3:
            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 

            CP_domain_vertices = self._cp._stable_domain_vertices
            for e in CP_domain_vertices:
#                 if e.name == 'Ba2CaV2CoF14':
                hull = ConvexHull(CP_domain_vertices[e])
                simplices = hull.simplices
                org_triangles = [CP_domain_vertices[e][s] for s in simplices]
                c=next(color)
                pc = a3.art3d.Poly3DCollection(org_triangles, alpha = alpha, facecolor=c,edgecolor = edc)
                ax.add_collection3d(pc)
                center = np.average(CP_domain_vertices[e], axis=0)
                print("center",center)
                text_font = {'fontname':'Arial', 'size':'15', 'color':c, 'weight':'normal'}
                ax.text(center[0],center[1],center[2],e.name,ha='center',
                        va='center',**text_font)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            ax.set_xlabel('Chem Pot '+ecopy[0],fontname='Arial',fontsize = 15)
            ax.set_ylabel('Chem Pot '+ecopy[1],fontname='Arial',fontsize = 15)
            ax.set_zlabel('Chem Pot '+ecopy[2],fontname='Arial',fontsize = 15)

        if self.sliceTitle == True:
            fixedStr = ''
            numStr = ''
            for e in self._cp.fixedEle:
                fixedStr += e[0] + " "
                numStr += str(e[1])+','
            if compound != None:
                title1 = "Slice of "+compound+" system with "+fixedStr+"mu fixed at " +numStr
            else:
                eleStr = "( "
                for ii in self._cp.elementList:
                    eleStr += ii +" "
                eleStr += ")" 
                title1 = "Slice of "+eleStr+" system with "+fixedStr+"mu fixed at " +numStr
            if title == "":
                title = title1
        plt.title(title, fontname = 'Arial', fontsize=15)
        plt.show()
        
    def get_chempot_plot(self,limits=None, compound = None,
                         title="",label_domains=True,edc = None, alpha = 0.1):
        """
        Plot Pourbaix diagram.

        Args:
            limits: 2D list containing limits of the Pourbaix diagram
                of the form [[xlo, xhi], [ylo, yhi]]
            title (str): Title to display on plot
            label_domains (bool): whether to label pourbaix domains
            plt (pyplot): Pyplot instance for plotting

        Returns:
            plt (pyplot) - matplotlib plot object with pourbaix diagram
        """
        if len(self.elementList) == 2:
            print(self.elementList)
            if limits is None:
                limits = [[-10, 0], [-10, 0]]
            
            plt.figure(figsize=(9.2, 7))
            
            xlim = limits[0]
            ylim = limits[1]
    
            label_font = {'fontname':'Calibri', 'size':'20', 'color':'black', 'weight':'normal'}
    #               'verticalalignment':'bottom'}
            text_font = {'fontname':'Calibri', 'size':'15', 'weight':'normal'}
            ax = plt.gca()
            ax.set_xlim(xlim)
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            ax.set_ylim(ylim)

            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            colorr=iter(jet(np.linspace(0,1,n)))
            cpentries = list(self._cp._stable_domain_vertices.keys())
            cpentries = sorted(cpentries,key = lambda e:e.entry.composition.get_atomic_fraction(self.elementList[0]))
            for entry in cpentries:
#                 if entry.name == 'BaN2':
#                 print(entry.name,vertices)
                vertices = self._cp._stable_domain_vertices[entry]
                center = np.average(vertices, axis=0)
                c=next(colorr)
                
                x, y = np.transpose(np.vstack([vertices, vertices[0]]))
#                     print(vertices)
#                     print(vertices[0])
#                     print(np.vstack([vertices, vertices[0]]))
#                     print(np.transpose(np.vstack([vertices, vertices[0]])))
                plt.fill(x, y, alpha = 0.2,facecolor = c)
    #             label_domains=False
                if label_domains:
                    plt.annotate(generate_entry_label(entry), center, ha='center',
                                 va='center', **text_font,color = c)
                    
            plt.xlabel('Chem Pot '+self.elementList[0],**label_font)
            plt.ylabel('Chem Pot '+self.elementList[1],**label_font)

#             plt.title(title, fontname = 'Arial', fontsize=20, fontweight='bold')
#             plt.title(title, fontname = 'Consolas', fontsize=15)
        if len(self.elementList) == 3:
            jet= plt.get_cmap('gist_rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            if limits is None:
                limits = [[-10, 0], [-10, 0], [-10, 0]]
            CP_domain_vertices = self._cp._stable_domain_vertices
            cpentries = list(self._cp._stable_domain_vertices.keys())
            cpentries = sorted(cpentries,key = lambda e:e.entry.composition.get_atomic_fraction(self.elementList[0]))
            for e in cpentries:
#                 if e.name == 'BaMnN2' or e.name == "Ba3MnN3":
                hull = ConvexHull(CP_domain_vertices[e])
                simplices = hull.simplices
#                 print(e.name)
#                 print(simplices)
#                 for s in simplices:
#                     print(s)
                org_triangles = [CP_domain_vertices[e][s] for s in simplices]
                c=next(color)
                pc = a3.art3d.Poly3DCollection(org_triangles, alpha = alpha, facecolor=c,edgecolor = edc)
                ax.add_collection3d(pc)
                center = np.average(CP_domain_vertices[e], axis=0)
#                 print(len(org_triangles),org_triangles[0])

#                 print(e.name,"vertices",CP_domain_vertices[e])
                print("center",center)
                text_font = {'fontname':'Calibri', 'size':'15', 'color':"k", 'weight':'normal'}
                ax.text(center[0],center[1],center[2],e.name,ha='center',
                        va='center',**text_font)

            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            ax.set_xticks(np.arange(limits[0][0],limits[0][1]+0.01,step = 0.5))
            ax.set_yticks(np.arange(limits[1][0],limits[1][1]+0.01,step = 0.5))
            ax.set_zticks(np.arange(limits[2][0],limits[2][1]+0.01,step = 0.5))
            ax.set_xlabel('Chem Pot '+self.elementList[0],fontname='Calibri',fontsize = 12)
            ax.set_ylabel('Chem Pot '+self.elementList[1],fontname='Calibri',fontsize = 12)
            ax.set_zlabel('Chem Pot '+self.elementList[2],fontname='Calibri',fontsize = 12)
#             plt.title(title, fontname = 'Consolas', fontsize=15,verticalalignment='center')
        if self.sliceTitle == True:
            fixedStr = ''
            numStr = ''
            for e in self._cp.fixedEle:
                fixedStr += e[0] + " "
                numStr += str(e[1])+','
            if compound != None:
                title1 = "Slice of "+compound+" system with "+fixedStr+"mu fixed at " +numStr
            else:
                eleStr = "( "
                for ii in self._cp.elementList:
                    eleStr += ii +" "
                eleStr += ")" 
                title1 = "Slice of "+eleStr+" system with "+fixedStr+"mu fixed at " +numStr
            if title == "":
                title = title1
        else:
            if compound != None:
                title1 = "Chem Pot Diagram of " + compound + "system"
            else:
                eleStr = "( "
                for ii in self.elementList:
                    eleStr += ii +" "
                eleStr += ")" 
                title1 = "Chem Pot Diagram of " + eleStr + "system"
            if title == "":
                title = title1
#         plt.title(title, fontname = 'Consolas', fontsize=15) #,verticalalignment='center')
        return plt
        
    def plot_entry_stability(self, entry, pH_range=None, pH_resolution=100,
                             V_range=None, V_resolution=100, e_hull_max=1,
                             cmap='RdYlBu_r', **kwargs):
        if pH_range is None:
            pH_range = [-2, 16]
        if V_range is None:
            V_range = [-3, 3]
        # plot the Pourbaix diagram
        plt = self.get_chempot_plot(**kwargs)
        pH, V = np.mgrid[pH_range[0]:pH_range[1]:pH_resolution * 1j,
                V_range[0]:V_range[1]:V_resolution * 1j]

        stability = self._cp.get_decomposition_energy(entry, pH, V)
        # Plot stability map
        plt.pcolor(pH, V, stability, cmap=cmap, vmin=0, vmax=e_hull_max)
        cbar = plt.colorbar()
        cbar.set_label("Stability of {} (eV/atom)".format(
            generate_entry_label(entry)))

        # Set ticklabels
        ticklabels = [t.get_text() for t in cbar.ax.get_yticklabels()]
        ticklabels[-1] = '>={}'.format(ticklabels[-1])
        cbar.ax.set_yticklabels(ticklabels)

        return plt

    def domain_vertices(self, entry):
        """
        Returns the vertices of the Pourbaix domain.

        Args:
            entry: Entry for which domain vertices are desired

        Returns:
            list of vertices
        """
        return self._cp._stable_domain_vertices[entry]
def compare_vertices(v1, v2, n1=0,n2=1):
    '''determine if 2 vertices x,y values are the same, if so, return True
        Args: v1, v2 : list of 3 number
          eg v1 = [1,1,1] '''

    if abs(v1[n1]-v2[n1])>1e-6 or abs(v1[n2]-v2[n2])>1e-6:
        return False # If the content is not equal, return false

    return True  
def get_edges_of_plane_in3D(vertices):
    '''
    Note that org_triangles, hull must comes from around vertices (around vertices)
    Project plane to lower dimension, then convex hull,
    simplices of 2D plane in 2D can tell us which is hyperplane (edge)'''
    two_vers_vertical = False
    print(vertices)
    for v1, v2 in combinations(vertices.tolist(),2):
        # print(v1,"\n",v2)
        if compare_vertices(v1,v2):
            two_vers_vertical = True
    
    xx = vertices[:,0].tolist()
    yy = vertices[:,1].tolist()
    zz = vertices[:,2].tolist()
    if len(list(set(yy))) == 1:
        n1=0
        n2=2
    elif len(list(set(xx))) == 1:
        n1=1
        n2=2
    elif len(list(set(zz))) == 1:
        n1=0
        n2=1
    elif not two_vers_vertical:
        n1=0
        n2=1
    else:
        n1=1
        n2=2


    print(n1,n2)
    lowD_vertices = np.hstack((vertices[:,n1].reshape((len(vertices),1)),
                               vertices[:,n2].reshape((len(vertices),1))))
    print(lowD_vertices)
    try:
        low_hull = ConvexHull(lowD_vertices)
    except:
        low_hull = ConvexHull(lowD_vertices, qhull_options='QJ')
    edges = [lowD_vertices[s].tolist() for s in low_hull.simplices]
    # print(np.array(edges))
    edges3D=[]
    for edge in edges:
        v1 = edge[0]
        v2 = edge[1]
        for vv in vertices.tolist():
            if abs(v1[0]-vv[n1])<1e-6 and abs(v1[1]-vv[n2])<1e-6:
                vv1=vv
            if abs(v2[0]-vv[n1])<1e-6 and abs(v2[1]-vv[n2])<1e-6:
                vv2=vv
        edges3D.append([vv1,vv2])
    # print(np.array(edges3D))
    return edges3D
def get_edges_of_polytope_in3D(org_triangles,hull):
    '''
    Note that org_triangles, hull must comes from original vertices (non-around vertices)
    And note that hull can not be QJ option
    find all lines of triangles; 
    then return lines that shared by 2 hyperplanes'''
    lines = []
    for tri in org_triangles:
        lines += list(combinations(tri.tolist(),2))
    # print(lines[0])
    lines1 = []
    for li in lines:
        v1=li[0]
        v2=li[1]
        num_planes_line_shared = 0

        eqs = hull.equations
        eqs = np.unique(eqs, axis = 0)
        for eq in eqs:
            
            if abs(np.dot(eq[0:3],v1)+eq[3]) <1e-6 \
                and abs(np.dot(eq[0:3],v2)+eq[3]) <1e-6:
                    num_planes_line_shared += 1
        
        if num_planes_line_shared == 2:
            '''the edge lines shared by two hyperplanes'''
            # print(abs(np.dot(eq[0:3],v1)+eq[3]),abs(np.dot(eq[0:3],v2)+eq[3]))
            if li not in lines1:
                lines1.append(li)
    return lines1
def generate_entry_label(entry):
    return entry.name

def overlap_lines_axis(lines):
    """
    Given a dict of 1-D lines as tuples of start and end points,
    returns a dict of lists, where each inner list corresponds to a segment
    of the x-axis and contains the indices of the lines that overlap with
    that segment.

    Args:
        lines (dict): a dict of tuples representing 1-D line segments

    Returns:
        dict: a list of lists, where each inner list contains the indices
        of the lines that overlap with a segment of the x-axis
    """
    # Create a list of all the unique x-axis values in the line segments
    x_values = set()
    for key, line in lines.items():
        x_values.add(line[0])
        x_values.add(line[1])

    # Sort the x-axis values
    x_values = sorted(x_values)

    # Create a dictionary mapping x-axis segments to the lines that overlap with them
    overlap_dict = {}
    for i in range(len(x_values) - 1):
        start = x_values[i]
        end = x_values[i+1]
        overlap_dict[(start, end)] = []
        for key, line in lines.items():
            if line[0] < end and line[1] > start:
                overlap_dict[(start, end)].append(key)

    return overlap_dict
def get_colors(entries, cmap = 'rainbow'):
    jet= plt.get_cmap(cmap)
    n=len(entries)
    color=iter(jet(np.linspace(0,1,n)))
    colordict = {}
    for e in entries:
        c = next(color)
        colordict[e.name] = c
    return colordict
def get_color_tri_array(points):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
     
    x = [1, 0, 0.5]
    y = [0, 0, math.sqrt(3) / 2]
    
    # barycentric coordinate
    xv1, xv2, xv3 = x[0], x[1], x[2]
    yv1, yv2, yv3 = y[0], y[1], y[2]
    
    px, py = points[:, 0], points[:, 1]
    denominator = (yv2 - yv3) * (xv1 - xv3) + (xv3 - xv2) * (yv1 - yv3)
    wv1 = ((yv2 - yv3) * (px - xv3) + (xv3 - xv2) * (py - yv3)) / denominator
    wv2 = ((yv3 - yv1) * (px - xv3) + (xv1 - xv3) * (py - yv3)) / denominator
    wv3 = 1 - wv1 - wv2

    c1, c2, c3 = np.array(colors[0]), np.array(colors[1]), np.array(colors[2])
    
    color = (wv1[:, np.newaxis] * c1 + wv2[:, np.newaxis] * c2 + wv3[:, np.newaxis] * c3) / (wv1[:, np.newaxis] + wv2[:, np.newaxis] + wv3[:, np.newaxis])
    return color
def get_color_tri(p):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      
    x = [1, 0, 0.5]
    y = [0, 0, math.sqrt(3) / 2]
     
    # barycentric coordinate
    xv1,xv2,xv3 = x[0],x[1],x[2]
    yv1,yv2,yv3 = y[0],y[1],y[2]
     
    px, py = p[0], p[1]
    denominator = (yv2-yv3)*(xv1-xv3) + (xv3-xv2)*(yv1-yv3)
    wv1 = ((yv2-yv3)*(px-xv3) + (xv3-xv2)*(py-yv3)) / denominator
    wv2 = ((yv3-yv1)*(px-xv3) + (xv1-xv3)*(py-yv3)) / denominator
    wv3 = 1 - wv1 - wv2
#     print(wv1, wv2, wv3)
    c1,c2,c3 = np.array(colors[0]), np.array(colors[1]), np.array(colors[2])
     
    color = (wv1*c1 + wv2*c2 + wv3*c3)/(wv1 + wv2 + wv3)
     
    return color
def calculate_plane_normal(point1, point2, point3):
    """
    calculate normal vector
    """
#     point1, point2, point3 = np.array(point1), np.array(point2), np.array(point3)
    vector1 = point2 - point1
    vector2 = point3 - point1

    normal_vector = np.cross(vector1, vector2)
    
    return normal_vector / np.linalg.norm(normal_vector)
def separate_points_by_plane(points, plane_normal, plane_point):
    """
    separate points by plane
    """
    points = np.array(points)
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)

    # calculate point to plane distance
    distances = np.dot(points - plane_point, plane_normal)

    # classify based on positive or negative distance
    on_plane = points[np.isclose(distances, 0)]
    on_positive_side = points[distances > 0]
    on_negative_side = points[distances < 0]

    return on_plane, on_positive_side, on_negative_side
