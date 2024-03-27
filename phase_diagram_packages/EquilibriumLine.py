# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import plotly.graph_objects as go
import logging

import itertools
import re
from copy import deepcopy
from functools import cmp_to_key, partial, lru_cache
from monty.json import MSONable, MontyDecoder

from multiprocessing import Pool
import warnings

from scipy.spatial import ConvexHull, HalfspaceIntersection
import numpy as np


try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from pymatgen.util.coord import Simplex
from pymatgen.util.string import latexify
from pymatgen.util.plotting import pretty_plot
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from pymatgen.core.ion import Ion
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter
from pymatgen.ext.matproj import MPRester
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




class EquilLine(MSONable):
    """
    Class to create a Equilibrium hyperplane on 
    chemical potential diagram from CPentries

    Args:
        entries (CPentries): Entries list
            containing Solids and Ions or a list of MultiEntries
        elementList: element str list of entries
        fixed:  list of tuples when a/mutiple element(s) chemical 
            potential is fixed at certain value. Eg: [("Fe",-1)]
        limits: boundary limits. 
            Eg: [[-10,0],[-10,0]] for a 2 components system
    """

    def __init__(self, entries, elementList,fixed = None, limits = None):
#         entries = deepcopy(entries)
        self._processed_entries = entries
        self.elementList = elementList
        self._stable_domain_vertices, self.intersection = \
            self.get_equil_line_domains(self._processed_entries,elementList,fixed = fixed,limits = limits)
        self.limits = limits
        if fixed != None:
            self.fixedEle = fixed.copy()
        else:
            self.fixedEle = fixed

    @staticmethod
    def get_equil_line_domains(CPentries, elementList, fixed = None, limits=None):
        """
        Return dual of convex hull in Chemical potential diagram.
        A more proper name should be equilibrium hyperplane.
        
        This function works by using scipy's HalfspaceIntersection
        function to construct all of the 2/3-D polygons that form the
        boundaries of the planes corresponding to chemical potential
        of elements. 
        
        Args:
            Same as EquilLine Class
            
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
                # N = number of entries C =  components number
                # 0 = G - muAxA - muBxB - muCxC
                # [xA, xB, xC] x [muA; muB; muC] = [G]
                # FOR HALFSPACEINTERSECTION, the vectors are
                # N x (C+1)  [xA, xB, xC, -G]
                
                hyperplane=[]
#                 print()
#                 print(entry.name,entry.ncomp)
                for z in range(0,C):
                    hyperplane.append(entry.ncomp[elementList[z]])
#                 hyperplane.append(1)
                hyperplane.append(-entry.form_E)
#                 print(entry.name)
#                 print(np.array(hyperplane))
                hyperplanes += [np.array(hyperplane)]

            hyperplanes = np.array(hyperplanes)
#             print(hyperplanes)

            # Add border hyperplanes and generate HalfspaceIntersection
            border_hyperplanes = []
            for j in range(2*C):
                border_hyperplane = []
                for i in range(C):
                    
                    if j == 2*i:
                        #sometimes you can change this to 0.8 or 0.9,
                        # and the phase can not visualize can show up again
                        border_hyperplane.append(-1)
                    elif j == 2*i + 1:
                        border_hyperplane.append(1)
                    else:
                        border_hyperplane.append(0)
#                 border_hyperplane.append(0)
                if (j%2) == 0:
                    border_hyperplane.append(limits[int(j/2)][0])
                else:
                    border_hyperplane.append(-limits[int((j-1)/2)][1])
#                 print(border_hyperplane)
                border_hyperplanes.append(border_hyperplane)

#             for bbb in border_hyperplanes:
#                 print(bbb)
#             print("border_hyperplanes",border_hyperplanes)
            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
#             print(hs_hyperplanes)
#             print(hs_hyperplanes[19])



            #You'll have to make the interior point N-dimensional as well.
            #  I Think if you fix limits to be N-dimensional, the interior point will also be 
            # (N+1)-dimensional, where the +1 is the energy dimension 
#             print(hs_hyperplanes)
            interior_point = np.average(limits, axis=1).tolist()
#             interior_point = [-0.1,-0.1]
            print("interior_point",interior_point)
            hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

            # organize the boundary points by entry
            EL_domain_vertices = {entry: [] for entry in CPentries}
#             print("haha", hs_int.intersections)
#             print("haha", hs_int.dual_facets)
            namelist = []
            for fa in hs_int.dual_facets:
                namelist.append([CPentries[ee].name for ee in fa if ee < len(CPentries)])

#                 print(fa, [CPentries[ee].name for ee in fa if ee < len(CPentries)])
#             print(len(CPentries))
#             for aaa in CPentries:
#                 if aaa.name == "O2":
#                     print(CPentries.index(aaa))
            for intersection, facet in zip(hs_int.intersections,
                                           hs_int.dual_facets):
                for v in facet:
                    if v < len(CPentries):
                        this_entry = CPentries[v]
                        EL_domain_vertices[this_entry].append(intersection)
                        
            # Remove entries with no pourbaix region
#             for k, v in EL_domain_vertices.items():
#                 if len(v) ==0:
#                     print(k.name)

            EL_domain_vertices = {k: v for k, v in EL_domain_vertices.items() if v}
            for ee in EL_domain_vertices:
                EL_domain_vertices[ee] = np.array(EL_domain_vertices[ee])

#                 if ee.name == "Cl2":
#                     print(EL_domain_vertices[ee])

            return EL_domain_vertices, hs_int.intersections

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
            # Create Hyperplanes
            # We're going to make it a variable length
            # The length of the hyperplane will be C + 1 - 1
            # N = number of entries, C = number of components
            # 0 = G - muAxA - muBxB - muCxC
            # if we fixed muB, then G-muBxB is a constant
            # [xA, xC] x [muA; muC] = [G-muBxB]
            # N x C, [xA; xC; G-muBxB]
            for entry in CPentries:
                
                hyperplane=[]
                for z in range(0,C):
                    if z not in fixIndex:
                        hyperplane.append(entry.ncomp[elementList[z]])

                formEMultiMux = 0
                for i in fixed:

                    formEMultiMux += i[1]*entry.ncomp[i[0]]
                formEMultiMux = formEMultiMux-entry.form_E
                hyperplane.append(formEMultiMux)
                print(entry.name,"hyperplane",hyperplane)
                hyperplanes += [np.array(hyperplane)]
            hyperplanes = np.array(hyperplanes)

            C = C-F

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

                if (j%2) == 0:
                     
                    border_hyperplane.append(limits[int(j/2)][0])
                else:
                    border_hyperplane.append(-limits[int((j-1)/2)][1])
#                 print(border_hyperplane)
                border_hyperplanes.append(border_hyperplane)

            print("border_hyperplanes",border_hyperplanes)
# 
            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
            print("hs_hyperplanes")
            print(hs_hyperplanes.tolist())
    

            #You'll have to make the interior point N-dimensional as well.
            #  I Think if you fix limits to be N-dimensional, the interior point will also be 
            # (N+1)-dimensional, where the +1 is the energy dimension 

            interior_point = np.average(limits, axis=1).tolist()
            print("interior_point")
            print(interior_point)


            hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))
#             print(hs_int.intersections)
            # organize the boundary points by entry
            EL_domain_vertices = {entry: [] for entry in CPentries}
            for intersection, facet in zip(hs_int.intersections,
                                           hs_int.dual_facets):
#                 print(intersection)
#                 print(facet)
                for v in facet:
                    if v < len(CPentries):
                        this_entry = CPentries[v]
                        EL_domain_vertices[this_entry].append(intersection)
                        
            # Remove entries with no pourbaix region
            EL_domain_vertices = {k: v for k, v in EL_domain_vertices.items() if v}
            for ee in EL_domain_vertices:
                EL_domain_vertices[ee] = np.array(EL_domain_vertices[ee])
#                 print(ee.name,EL_domain_vertices[ee])

            return EL_domain_vertices, hs_int.intersections




    @property
    def stable_entries(self):
        """
        Returns the stable entries in the Pourbaix diagram.
        """
        return list(self._stable_domain_vertices.keys())

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

    @classmethod
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


class EquilLinePlotter:
    """
    A plotter class for dual of convex hull in CP diagram.

    Args:
        EquilLine: A EquilLine object.
        elementList: Whether unstable phases will be plotted as well as
            red crosses. Defaults to False.
    """

    def __init__(self, EquilLine, elementList = None):
        self._eql = EquilLine
        self.fixed = EquilLine.fixedEle
        if elementList == None:
            self.elementList = EquilLine.elementList
        if self.fixed != None:
            fixe = [f[0] for f in self.fixed]
            unfixelementList = []
            for i in self.elementList:
                if i not in fixe:
                    unfixelementList.append(i)
            self.unfixelementList = unfixelementList

            
    def get_equilLine_plot(self,limits=None,
                         title="",label_domains=True,edc = None, alpha = 0.1):
        

        plotE = self.elementList
        if self.fixed != None:
            plotE = self.unfixelementList
        if len(plotE) == 2:
            print(self.elementList)
            if limits is None:
                limits = [[-10, 0], [-10, 0]]
            fig = plt.figure(figsize=(9.2, 7))
 
            ax = plt.gca()
            vertices = self._eql.intersection
            hull = ConvexHull(vertices)
            plt.plot(vertices[:,0], vertices[:,1], 'o')
            for simplex in hull.simplices:
                plt.plot(vertices[simplex,0],vertices[simplex,1],'k-')
            text_font = {'fontname':'Consolas', 'size':'15', 'weight':'normal'}

            for entry, vertices in self._eql._stable_domain_vertices.items():
                center = np.average(vertices, axis=0)
                if label_domains:
                    plt.annotate(generate_entry_label(entry), center, ha='center',
                                 va='center', **text_font) 
            ax.set_xlim(limits[0])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])
  
            plt.show()

        if len(plotE) == 3:
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            hull = ConvexHull(self._eql.intersection)
            simplices = hull.simplices
            org_triangles = [self._eql.intersection[s] for s in simplices]
            pc = a3.art3d.Poly3DCollection(org_triangles, alpha = 0.3,edgecolor = 'w')
            ax.add_collection3d(pc)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            if limits is None:
                limits = [[-10, 0], [-10, 0], [-10, 0]]
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            if plotE == self.elementList:
                ax.set_xlabel('Chem Pot '+self.elementList[0],fontname='Consolas',fontsize = 12)
                ax.set_ylabel('Chem Pot '+self.elementList[1],fontname='Consolas',fontsize = 12)
                ax.set_zlabel('Chem Pot '+self.elementList[2],fontname='Consolas',fontsize = 12)
            text_font = {'fontname':'Consolas', 'size':'15', 'weight':'normal'}

            for cpdata in self._eql.intersection:
                ax.scatter(cpdata[0],cpdata[1],cpdata[2],c = 'k',s = 30)
            for entry, vertices in self._eql._stable_domain_vertices.items():
                center = np.average(vertices, axis=0)
                print(center)
                if label_domains:
                    ax.text(center[0],center[1],center[2],entry.name,ha='center',
                        va='center',**text_font) 
            plt.show()
        
def generate_entry_label(entry):
    return entry.name
