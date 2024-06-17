# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import plotly.graph_objects as go
import logging
from functools import cmp_to_key
from phase_diagram_packages.ChemicalPotentialDiagram import trans_PD_to_ChemPot_entries, ChemPotEntry
from phase_diagram_packages.EquilibriumLine import EquilLine
import itertools
from monty.json import MSONable, MontyDecoder

from scipy.spatial import ConvexHull, HalfspaceIntersection
import numpy as np
from itertools import combinations
from adjustText import adjust_text
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from pymatgen.util.coord import Simplex
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter
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
# MU_H2O = -2.6583
PREFAC = 0.0591
ELEMENTS = set(['H','He','Li','Be','B','C','N','O','F','Ne',
           'Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca',
           'Sc', 'Ti', 'V','Cr', 'Mn', 'Fe', 'Co', 'Ni',
           'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
           'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
           'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
           'I', 'Xe','Cs', 'Ba','La', 'Ce', 'Pr', 'Nd', 'Pm',
           'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
           'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
           'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
           'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
           'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
           'Rf', 'Db', 'Sg', 'Bh','Hs', 'Mt', 'Ds', 'Rg', 'Cn',
           'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])
class GeneralizedEntry(MSONable):

    def __init__(self, name,formEperformula, elslist,
                 normalized_els = None, 
                 surf_product=0,
                 entry_id = None,
                 q=0):
        self.form_E = formEperformula
        self.name = name
        self.entry_id = entry_id
        # print(name)
        self.q=q
        if "^" in name:
            q = name.split("^")[-1][1:-1]
            if q[-1] == "-":
                self.q = int(q[:-1])*(-1)
            else:
                self.q = int(q[:-1])
#         print(self.q)
        pureformula = name.split("^")[0]
        symbol = ""
        if "-" in pureformula:
            pureformula = pureformula.split("-")
            symbol = pureformula[0]
            if symbol == "R":
                symbol = ""
            pureformula = pureformula[1]
        else:
            pureformula = pureformula.split("-")[-1]
        
        elmap = {i:0 for i in elslist}
        self.pureformula = pureformula
        els_seq = []
        for i in Composition(pureformula)._data:
            els_seq.append(str(i))
            elmap[str(i)]=Composition(pureformula)._data[i]
        self.elmap = elmap

        self.elslist = elslist
        self.Nt = sum(elmap.values())
        if normalized_els:
            self.Ne = 0
            for el in normalized_els:
                self.Ne += elmap[el]
        
        if symbol:
            symbol = r'$\%s$' % symbol + "-"


        for el in els_seq:
            if self.elmap[el] != 0:
                if self.elmap[el] != 1:
                    if self.elmap[el] != int(self.elmap[el]):
                        symbol += r"$%s_{%.2f}$" % (el, self.elmap[el])
                    else:
                        symbol += r"$%s_{%d}$" % (el, self.elmap[el])
                else:
                    symbol += el
        if self.q != 0:
            if abs(self.q) != 1:
                if self.q > 0:
                    symbol += r"$^{%s}$" % (str(self.q) + "+")
                if self.q < 0:
                    symbol += r"$^{%s}$" % (str(abs(self.q)) + "-")
            else:
                if self.q > 0:
                    symbol += r"$^{%s}$" % ("+")
                if self.q < 0:
                    symbol += r"$^{%s}$" % ("-")             
        self.symbol = symbol
        if "Mn" in elslist:
            self.Nm = elmap["Mn"]
        elif "Ga" in elslist:
            self.Nm = elmap["Ga"]
        '''surf_product is the product of surface energy J/m^2, 
        shape factor, volume/Mn angstrom^3/atom'''
        self.surf_product = surf_product
        # print(self.surf_product,"\n")
    def __str__(self):
        return "Generalized Entry, name: {}, formE per formula: {}".format(self.name, self.form_E)
        
         
        
        
        
        
class GeneralizedDiagram(MSONable):
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

    def __init__(self, entries,elementList,fixed = None, mutoPH=False, limits = None, 
                 normalizedfactor="Nt",slicePlane=False,sliceLower=False,surf=False,
                 w_aqu_constraint = False, extra_comp_axis = None,
                 specific_muh2o = None,
                 selfslicelist = None,
                 ammonia = None,
                 temperature = 298.15):
#         entries = deepcopy(entries)
        self._processed_entries = entries
        self.mutoPH = mutoPH
        self.elementList = elementList
        self.limits = limits
        self.slicePlane = slicePlane
        self.surf = surf
        self.extra_comp_axis = extra_comp_axis
        
        '''new h2o formation energy at different t and p'''
        if not specific_muh2o:
            muh2o = MU_H2O
        else:
            muh2o = specific_muh2o
        self.muh2o = muh2o
        self.ammonia = ammonia
        if temperature == 298.15:
            self.prefac = PREFAC
        else:
            self.prefac = PREFAC / 298.15 * temperature
        if slicePlane:
            if not selfslicelist:
                if not ammonia:
                    suffd = {el:0 for el in elementList}
                    suffd["O"]=1
                    suffd["H"]=2*(-self.prefac)
                    if not mutoPH:
                        suffd["H"]=2
                    # mu_O+2mu_(H+)-2E+0*phi+(-mu_H2O-mu_Mn*Nmn)=0, mu_Mn=0
                    fixelements = []
                    if fixed != None:
                        for i in fixed:
                            fixelements.append(i[0])
                    slicelist = []
                    for el in elementList:
                        if el not in fixelements:
                            slicelist.append(suffd[el])
                    
                    if not surf:
                        slicelist += [-2,0,-muh2o]
                    else:
                        slicelist += [-2,0,0,-muh2o]
                    
                    self.slicelist = slicelist
                    print(slicelist)

                else:
                    suffd = {el:0 for el in elementList}
                    suffd["N"]=1
                    suffd["H"]=3*(-self.prefac)
                    if not mutoPH:
                        suffd["H"]=3
                    # mu_N+3mu_(H+)-3E+0*phi+(-mu_NH3-mu_Mn*Nmn)=0, mu_Mn=0
                    fixelements = []
                    if fixed != None:
                        for i in fixed:
                            fixelements.append(i[0])
                    slicelist = []
                    for el in elementList:
                        if el not in fixelements:
                            slicelist.append(suffd[el])
                    
                    if not surf:
                        slicelist += [-3,0,-muh2o]
                    else:
                        slicelist += [-3,0,0,-muh2o]
                    print(slicelist)

                    self.slicelist = slicelist
            else:
                self.slicelist = selfslicelist
            self.sliceLower = sliceLower

        self.normalizedfactor = normalizedfactor
        if not w_aqu_constraint:
            self._stable_domains, self._stable_domain_vertices= \
                self.get_generalized_chem_pot_domains(self._processed_entries,elementList,fixed = fixed,mutoPH=mutoPH,limits = limits)
        if w_aqu_constraint:
            self._stable_domains1, self._stable_domain_vertices1= \
                self.get_generalized_chem_pot_domains_w_aqueous_constraints(self._processed_entries,
                                                                            elementList,
                                                                            fixed = fixed,
                                                                            mutoPH=mutoPH,
                                                                            limits = limits,
                                                                            extra_comp_axis = extra_comp_axis)
        
        if fixed != None:
            self.fixedEle = fixed.copy()
        

        
        

    
    def get_generalized_chem_pot_domains(self,CPentries, elementList, fixed = None, mutoPH=False,phi = None,limits=None):
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
        if "H" in elementList:
            hindex = elementList.index("H")

        # Get hyperplanes
        hyperplanes = []
        if fixed == None:
            if limits is None:
                limits = []
                for i in range(len(elementList)):
                    limits += [[-10,0]]
                if self.surf:
                    limits += [[0,1]]
                '''for q'''
                limits += [[-5,5]]

            for entry in CPentries:
                # Create Hyperplanes
                # We're going to make it a variable length
                # The length of a hyperplane will be C + 4, where C = number of components
                # N = number of entries
                # Phi = 1/Nt* (G_formula - EQ - muA*NA - muB*NB - muC*NC) + 1/R*surf_product
            
                # [NA, NB, NC, Q, -Nt*surf_p, Nt] x [muA; muB; muC; E; 1/R; Phi] = [G_formula]
                #  Nx(C+3) matrix        (C+3)x1 matrix    Nx1
                # FOR HALFSPACEINTERSECTION, the vectors are
                # N x (C+4)
                # [NA, NB, NC, Q, -Nt*surf_p, Nt, -G]
                if self.normalizedfactor == "Nt":
                    normalized = entry.Nt
                elif self.normalizedfactor == "Ne":
                    normalized = entry.Ne
                else:
                    normalized = entry.Nm
                hyperplane=[]
                for z in range(0,C):
                    hyperplane.append(entry.elmap[elementList[z]])
                
                if self.surf:
                    hyperplane += [entry.q, -entry.surf_product*normalized, normalized, -entry.form_E]
                else:
                    hyperplane += [entry.q, normalized, -entry.form_E]
                if mutoPH == True:
                    hyperplane[hindex]=hyperplane[hindex]*(-self.prefac)
                    limits[hindex]=[0,14]
                    hyperplane[C] -= entry.elmap["H"] #E
#                 print(hyperplane)

                hyperplanes += [np.array(hyperplane)]
                print(entry.name)
                print(hyperplane)
            hyperplanes = np.array(hyperplanes)


            max_contribs = np.max(np.abs(hyperplanes), axis=0)
            g_maxList = [] #limits[0][1], limits[1][1], 0, 1
            for i in range(C+1):
                g_maxList.append(limits[i][1])
            if self.surf:
                g_maxList.append(1) # for surface energy
            g_maxList.append(0) # for phi
            g_maxList.append(1) # for formE
            # print(g_maxList)
            g_max = np.dot(-max_contribs, g_maxList)

#             # Add border hyperplanes and generate HalfspaceIntersection
#             ## TODO: Now we have to make the border_hyperplanes also N-dimensional. 
#             ## You will also have to respecify the variable 'limits' to be N-dimensional. 

            border_hyperplanes = []
            for j in range(2*(len(limits))):
                border_hyperplane = []
                for i in range(len(limits)):
                     
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
                # print(border_hyperplane)
                border_hyperplanes.append(border_hyperplane)
            border_hyperplane = []

            for i in range(len(limits)):
                border_hyperplane.append(0)
            border_hyperplane.append(-1)
            border_hyperplane.append(2*g_max)
            # print(border_hyperplane)
            border_hyperplanes.append(border_hyperplane)
            # print("border_hyperplanes","\n",np.array(border_hyperplanes))

            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])

            interior_point = np.average(limits, axis=1).tolist() + [g_max]
#             print(interior_point)
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

            CPdomains = {k: v for k, v in CPdomains.items() if v}
            CP_domain_vertices = {}
#             CP_volume = {}
            if phi != None:
                for entry, points in CPdomains.items():
                    points = np.array(points)[:,:len(limits) + 1]
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
                # print(entry.name)
                points = np.array(points)[:,:len(limits)]
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
#                 CP_volume[entry] = hull.volume
                simplices = [points[indices] for indices in hull.simplices]
                CPdomains[entry] = simplices
                CP_domain_vertices[entry] = points

                
        else:
            F = len(fixed)
            fixIndex = [elementList.index(fixed[i][0]) for i in range(F)]
            if "H" in elementList:
                org_hindex = hindex
                for i in fixIndex:
                    if i < org_hindex:
                        hindex -= 1

            if limits is None:
                limits = []
                for i in range(C-F):
                    limits += [[-10,0]]
                limits += [[-5,5]]
                if self.surf:
                    limits += [[0,1]]
            else:
                newlimits = []
                for i in range(len(limits)):
                    if i not in fixIndex:
                        newlimits += [limits[i]]
                limits = newlimits

            for iiii in range(F):
                print("chemical potential of element",fixed[iiii][0]," is fixed at",fixed[iiii][1])

            for entry in CPentries:
                # Create Hyperplanes, C: number of components
                # We're going to make it a variable length
                # N = number of entries
                # Phi = 1/Nt* (G_formula - EQ - muA*NA - muB*NB - muC*NC) + 1/R*surf_product
                #  Nx(C+2) matrix        (C+2)x1 matrix    Nx1
                # [NB, NC, Q, Nt*surf_p, Nt] x [muB; muC; E; 1/R; Phi] = [G_formula-muA*NA]
                
                # FOR HALFSPACEINTERSECTION, the vectors are C+4-len(fixed)
                # [NB, NC, Q, Nt*surf_p, Nt, -G+sum(all fix muA*NA)]
                # print(entry.name)
                if self.normalizedfactor == "Nt":
                    normalized = entry.Nt
                elif self.normalizedfactor == "Ne":
                    normalized = entry.Ne
                else:
                    normalized = entry.Nm

                hyperplane=[]

                for z in range(0,C):
                    if z not in fixIndex:
                        hyperplane.append(entry.elmap[elementList[z]])
                if self.surf:
                    hyperplane += [entry.q,-entry.surf_product*normalized,normalized]
                else:
                    hyperplane += [entry.q,normalized]
                formEMultiMux = 0
                for i in fixed:
                    formEMultiMux += i[1]*entry.elmap[i[0]]
                formEMultiMux = formEMultiMux-entry.form_E
                hyperplane.append(formEMultiMux)
#                 print(entry.name)
#                 print(hyperplane)
                if mutoPH == True:
                    hyperplane[hindex]=hyperplane[hindex]*(-self.prefac)
                    limits[hindex]=[0,14]
                    hyperplane[C-F] -= entry.elmap["H"]
                
#                 print("hyperplane",hyperplane)
                hyperplanes += [np.array(hyperplane)]
            
            if self.slicePlane:
                slicelist = self.slicelist
                if self.sliceLower:
                    slicelist = [i*(-1) for i in self.slicelist]
                hyperplanes += [np.array(slicelist)]
            
            hyperplanes = np.array(hyperplanes)
            
            #########################################################
            C = C-F
            # print(np.abs(hyperplanes))
            max_contribs = np.max(np.abs(hyperplanes), axis=0)
            # print(max_contribs)
            # max_contribs = np.array([4,0.1773,4,0,3,13.3])

            g_maxList = [] #limits[0][1], limits[1][1], 0, 1
            for i in range(C+1):
                g_maxList.append(limits[i][1])
            if self.surf:
                g_maxList.append(1) # for surface energy
            g_maxList.append(0) # for phi
            g_maxList.append(1) # for formE
            # print(g_maxList)
            g_max = np.dot(-max_contribs, g_maxList)
#             print()
#             print(g_max)
#             print(limits)
    
            # Add border hyperplanes and generate HalfspaceIntersection
            
            ## TODO: Now we have to make the border_hyperplanes also N-dimensional. 
            ## You will also have to respecify the variable 'limits' to be N-dimensional. 
            border_hyperplanes = []
            for j in range(2*(len(limits))):
                border_hyperplane = []
                for i in range(len(limits)):
                     
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
                # print(border_hyperplane)
                border_hyperplanes.append(border_hyperplane)
            border_hyperplane = []

            for i in range(len(limits)):
                border_hyperplane.append(0)
            border_hyperplane.append(-1)
            border_hyperplane.append(2*g_max)
            # print(border_hyperplane)
            border_hyperplanes.append(border_hyperplane)
            # print("border_hyperplanes","\n",np.array(border_hyperplanes))

            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
#             print(hs_hyperplanes)
    
            #You'll have to make the interior point N-dimensional as well.
            #  I Think if you fix limits to be N-dimensional, the interior point will also be 
            # (N+1)-dimensional, where the +1 is the energy dimension 
            
            interior_point = np.average(limits, axis=1).tolist() + [g_max]
#             print(interior_point)
#             interior_point[-1] = -130
            if self.slicePlane and self.sliceLower:
                interior_point[-2] = -1

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
#             CP_volume = {}
            if len(fixed) == len(elementList)-1:
                for entry, points in CPdomains.items():
#                     points = np.array(points)[:,:C+1]
#                     CPdomains[entry] = points
                    points = np.array(points)[:,:len(limits)]
                    CP_domain_vertices[entry] = points
#                     CP_volume[entry] = None
            else:
                print()
                for entry, points in CPdomains.items():
                    print(entry.name)
    #                 if entry.name == 'Ba4NaAl2B8(ClO6)3':
                    points = np.array(points)[:,:len(limits)]
                    # Initial sort to ensure consistency
                    points = points[np.lexsort(np.transpose(points))]
                    # print('P2', points)
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
                    hull = ConvexHull(points,qhull_options='QJ')
#                     CP_volume[entry] = hull.volume
                    simplices = [points[indices] for indices in hull.simplices]
    #                 print(simplices)
                    CPdomains[entry] = simplices
                    CP_domain_vertices[entry] = points
                    # print(points,"\n")
        return CPdomains, CP_domain_vertices

    def get_generalized_chem_pot_domains_w_aqueous_constraints(self,CPentries, elementList, fixed = None, mutoPH=False,
                                                               phi = None,limits=None,show_phi=False,
                                                               extra_comp_axis = None):
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
        print(elementList)
        if "H" in elementList:
            hindex = elementList.index("H")
        # Get hyperplanes
        hyperplanes = []
        if self.ammonia:
            erase_el = "N"
            coefficient = 3
        else:
            erase_el = "O"
            coefficient = 2
        if fixed != None:
            fix1 = [i for i in fixed if i[0] in elementList]
            fixed2 = [i for i in fixed if i[0] not in elementList]
            fixVar = {ee:val for ee,val in fixed2}
            print(fixed2)
            print(fixVar)
            fixed = fix1
            F = len(fixed)
            fixIndex = [elementList.index(fixed[i][0]) for i in range(F)]
            hindex_o = hindex
            for i in fixIndex:

                if i < hindex_o:
                    hindex -= 1
            if elementList.index(erase_el) < elementList.index("H"):
                hindex -= 1
            if limits is None:
                limits = []
                for i in range(C-F-1):
                    limits += [[-10,0]]
                limits += [[-5,5]]
                if self.surf:
                    limits+=[[0,1]]
            else:
                newlimits = []
                for i in range(C):
                    if i not in fixIndex and elementList[i] != erase_el:
                        newlimits += [limits[i]]
                if len(fixed2) == 0:
                    newlimits += limits[C:]
                else:
                    if extra_comp_axis == None or len(extra_comp_axis) == 0:
                        if "E" not in fixVar:
                            newlimits += [limits[-2]]
                        if "1/R" not in fixVar:
                            newlimits += [limits[-1]]
                    else:
                        if "E" not in fixVar:
                            newlimits += [limits[-1 - len(extra_comp_axis)]]
                        for i, els in enumerate(extra_comp_axis):
                            if "log"+els not in fixVar:
                                newlimits += [limits[-(len(extra_comp_axis)-i)]]                        
            
                limits = newlimits
            print(limits)

            for iiii in range(F):
                print("chemical potential of element",fixed[iiii][0],"is fixed at",fixed[iiii][1])
            

            for entry in CPentries:
                # WITHOUT EXTRA ION AXIS
                # Create Hyperplanes
                # We're going to make it a variable length
                # N = number of entries
                # muB = MU_H2O - 2*muC + 2*E
                # Phi = 1/Nt* (G_formula - EQ - muA*NA - muB*NB - muC*NC) + 1/R*surf_product
                #  Nx(C+1) matrix        (C+1)x1 matrix    Nx1
                # A:metal,B:O,C:H+ (not H) 
                # [NC-2*NB, Q-NC+2*NB, Nt*surf_p, Nt] x [muC; E; 1/R; Phi] = [G_formula-muA*NA-MU_H2O*NB]
                
                # FOR HALFSPACEINTERSECTION, the vectors are 
                # N x (C+4-len(fixed)-1), the final 1 is from muO water equilibrium
                # [NC-2*NB, Q-NC+2*NB, -Nt*surf_p, Nt, -G+sum(all fix muA*NA)+MU_H2O*NB]
                
                # WITH EXTRA ION AXIS
                # [NC-2*NB, Q-NC+2*NB, -Nt*surf_p, [-RT]*len(extra_ion_axis), Nt, -G+sum(all fix muA*NA)+MU_H2O*NB]
                if self.normalizedfactor == "Nt":
                    normalized = entry.Nt
                elif self.normalizedfactor == "Ne":
                    normalized = entry.Ne
                else:
                    normalized = entry.Nm

                hyperplane=[]
                
                for z in range(0,C):
                    if elementList[z] == "H":
                        hyperplane.append(entry.elmap[elementList[z]]-coefficient*entry.elmap[erase_el])
                    elif z not in fixIndex and elementList[z] != erase_el:
                        hyperplane.append(entry.elmap[elementList[z]])

                if "E" not in fixVar:
                    hyperplane += [entry.q + coefficient*entry.elmap[erase_el], normalized]
                else:
                    hyperplane += [normalized]
                surface_index = len(hyperplane) - 1
                if self.surf:
                    hyperplane.insert(surface_index,-normalized*entry.surf_product)
                formEMultiMux = 0
                for i in fixed:
                    formEMultiMux += i[1]*entry.elmap[i[0]]
                
                formEMultiMux = formEMultiMux - entry.form_E + self.muh2o * entry.elmap[erase_el]
                if "E" in fixVar:
                    formEMultiMux += fixVar["E"] * (entry.q + coefficient * entry.elmap[erase_el])
                if "1/R" in fixVar:
                    formEMultiMux -= normalized * entry.surf_product * fixVar["1/R"]
                hyperplane.append(formEMultiMux)
                # print(hyperplane)
                # print(hindex)
                if mutoPH == True:
                    hyperplane[hindex]=hyperplane[hindex]*(-self.prefac)
                    limits[hindex]=[0,14]
                    if "E" not in fixVar:
                        hyperplane[C-F-1] -= entry.elmap["H"] #E
                    else:
                        hyperplane[-1] -= entry.elmap["H"] * fixVar["E"]

                if extra_comp_axis:
                    for ell in extra_comp_axis:
                        print(entry.name, ell, entry.q == 0 or ell not in entry.name)
                        if entry.q == 0 or ell not in entry.name:
                            if "log" + ell not in fixVar:
                                hyperplane.insert(-2, 0)
                                print(hyperplane)
                        else:
                            if "log" + ell not in fixVar:
                                hyperplane.insert(-2, -self.prefac)
                            else:
                                print("log" + ell, fixVar["log" + ell],self.prefac * fixVar["log" + ell])
                                hyperplane[-1] -= self.prefac * fixVar["log" + ell]
                            
                print(entry.name, "extra", hyperplane)
                hyperplanes += [np.array(hyperplane)]


            hyperplanes = np.array(hyperplanes)
            #########################################################
            C = C-F
            max_contribs = np.max(np.abs(hyperplanes), axis=0)
            # print(max_contribs)
            g_maxList = [] #limits[0][1], limits[1][1], 0, 1
            for i in range(len(limits)):
                g_maxList.append(limits[i][1])
            g_maxList.append(0)
            g_maxList.append(1)
            g_max = np.dot(-max_contribs, g_maxList)
            # print()
            # print(g_max)
            # print(limits)
            # Add border hyperplanes and generate HalfspaceIntersection
            
            ## TODO: Now we have to make the border_hyperplanes also N-dimensional. 
            ## You will also have to respecify the variable 'limits' to be N-dimensional. 
            border_hyperplanes = []
            for j in range(2*(len(limits))):
                border_hyperplane = []
                for i in range(len(limits)):
                     
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
                # print(border_hyperplane)
                border_hyperplanes.append(border_hyperplane)
            border_hyperplane = []
            for i in range(len(limits)):
                border_hyperplane.append(0)
            border_hyperplane.append(-1)
            border_hyperplane.append(2*g_max)
            # print(border_hyperplane)
            border_hyperplanes.append(border_hyperplane)
            
            # print("border_hyperplanes",border_hyperplanes)
            
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
#             CP_volume = {}
            for entry, points in CPdomains.items():
                print(entry.name)
#                 points: PH,(other els),E,(1/R),PHI len(limits)+1
                if show_phi:
                    points = np.array(points)[:,:len(limits)+1] #+1 is for phi values
                else:
                    points = np.array(points)[:,:len(limits)]
                # Initial sort to ensure consistency
                points = points[np.lexsort(np.transpose(points))]
                center = np.average(points, axis=0)
                points_centered = points - center 
                point_comparator = lambda x, y: x[0] * y[1] - x[1] * y[0]
                points_centered = sorted(points_centered,
                                         key=cmp_to_key(point_comparator))
                points = points_centered + center
                if len(points) < 4:
                    points = np.vstack((points, points[0]))
                # print(points)
                hull = ConvexHull(points,qhull_options='QJ')
#                     CP_volume[entry] = hull.volume
                # for indices in hull.simplices:
                #     print(indices)
                #     print(points[indices])
                simplices = [points[indices] for indices in hull.simplices]

                CPdomains[entry] = simplices
                CP_domain_vertices[entry] = points
                print(points)


        
        return CPdomains, CP_domain_vertices
    
    def get_generalized_chem_pot_domains_w_aqueous_constraints2(self,CPentries, elementList, fixed = None, mutoPH=False,
                                                               phi = None,limits=None,show_phi=False):
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
        print(elementList)
        if "H" in elementList:
            hindex = elementList.index("H")
        # Get hyperplanes
        hyperplanes = []
        if fixed != None:
            fix1 = [i for i in fixed if i[0] in elementList]
            fixed2 = [i for i in fixed if i[0] not in elementList]
            fixVar = {ee:val for ee,val in fixed2}
            fixed = fix1
            F = len(fixed)
            fixIndex = [elementList.index(fixed[i][0]) for i in range(F)]
            hindex_o = hindex
            for i in fixIndex:

                if i < hindex_o:
                    hindex -= 1
            if elementList.index("O") < elementList.index("H"):
                hindex -= 1
            if limits is None:
                limits = []
                for i in range(C-F-1):
                    limits += [[-10,0]]
                limits += [[-5,5]]
                if self.surf:
                    limits+=[[0,1]]
            else:
                newlimits = []
                for i in range(C):
                    if i not in fixIndex and elementList[i] != "O":
                        newlimits += [limits[i]]
                if len(fixed2) == 0:
                    newlimits += limits[C:]
                else:
                    
                    if "E" not in fixVar:
                        newlimits += [limits[-2]]
                    if "1/R" not in fixVar:
                        newlimits += [limits[-1]]
                limits = newlimits
            print(limits)

            for iiii in range(F):
                print("chemical potential of element",fixed[iiii][0],"is fixed at",fixed[iiii][1])

            for entry in CPentries:
                # WITHOUT EXTRA ION AXIS
                # Create Hyperplanes
                # We're going to make it a variable length
                # N = number of entries
                # muB = MU_H2O - 2*muC + 2*E
                # Phi = 1/Nt* (G_formula - EQ - muA*NA - muB*NB - muC*NC) + 1/R*surf_product
                #  Nx(C+1) matrix        (C+1)x1 matrix    Nx1
                # A:metal,B:O,C:H+ (not H) 
                # [NC-2*NB, Q-NC+2*NB, Nt*surf_p, Nt] x [muC; E; 1/R; Phi] = [G_formula-muA*NA-MU_H2O*NB]
                
                # FOR HALFSPACEINTERSECTION, the vectors are 
                # N x (C+4-len(fixed)-1), the final 1 is from muO water equilibrium
                # [NC-2*NB, Q-NC+2*NB, -Nt*surf_p, Nt, -G+sum(all fix muA*NA)+MU_H2O*NB]
                
                # WITH EXTRA ION AXIS
                # [NC-2*NB, Q-NC+2*NB, -Nt*surf_p, [-RT]*len(extra_ion_axis), Nt, -G+sum(all fix muA*NA)+MU_H2O*NB]
                if self.normalizedfactor == "Nt":
                    normalized = entry.Nt
                elif self.normalizedfactor == "Ne":
                    normalized = entry.Ne
                else:
                    normalized = entry.Nm
                hyperplane=[]

                for z in range(0,C):
                    if elementList[z] == "H":
                        hyperplane.append(entry.elmap[elementList[z]]-2*entry.elmap["O"])
                    elif z not in fixIndex and elementList[z] != "O":
                        hyperplane.append(entry.elmap[elementList[z]])

                if "E" not in fixVar:
                    hyperplane += [entry.q+2*entry.elmap["O"], normalized]
                else:
                    hyperplane += [normalized]
                surface_index = len(hyperplane) - 1
                if self.surf:
                    hyperplane.insert(surface_index,-normalized*entry.surf_product)
                formEMultiMux = 0
                for i in fixed:
                    formEMultiMux += i[1]*entry.elmap[i[0]]

                formEMultiMux = formEMultiMux-entry.form_E+MU_H2O*entry.elmap["O"]
                if "E" in fixVar:
                    formEMultiMux += fixVar["E"] * (entry.q+2*entry.elmap["O"])
                if "1/R" in fixVar:
                    formEMultiMux -= normalized * entry.surf_product * fixVar["1/R"]
                hyperplane.append(formEMultiMux)
                # print(hyperplane)
                # print(hindex)
                if mutoPH == True:
                    hyperplane[hindex]=hyperplane[hindex]*(-self.prefac)
                    limits[hindex]=[0,14]
                    if "E" not in fixVar:
                        hyperplane[C-F-1] -= entry.elmap["H"] #E
                    else:
                        hyperplane[-1] -= entry.elmap["H"] * fixVar["E"]

                if self.extra_comp_axis:
                    for ell in self.extra_comp_axis:
                        print(entry.name, ell, entry.q == 0 or ell not in entry.name)
                        if entry.q == 0 or ell not in entry.name:
                            hyperplane.insert(-2, 0)
                        else:
                            hyperplane.insert(-2, -self.prefac)
                            
                # print(entry.name,hyperplane)
                hyperplanes += [np.array(hyperplane)]
            # gaga
            hyperplanes = np.array(hyperplanes)
            #########################################################
            C = C-F
            max_contribs = np.max(np.abs(hyperplanes), axis=0)
            # print(max_contribs)
            g_maxList = [] #limits[0][1], limits[1][1], 0, 1
            for i in range(len(limits)):
                g_maxList.append(limits[i][1])
            g_maxList.append(0)
            g_maxList.append(1)
            g_max = np.dot(-max_contribs, g_maxList)
            # print()
            # print(g_max)
            # print(limits)
            # Add border hyperplanes and generate HalfspaceIntersection
            
            ## TODO: Now we have to make the border_hyperplanes also N-dimensional. 
            ## You will also have to respecify the variable 'limits' to be N-dimensional. 
            border_hyperplanes = []
            for j in range(2*(len(limits))):
                border_hyperplane = []
                for i in range(len(limits)):
                     
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
                # print(border_hyperplane)
                border_hyperplanes.append(border_hyperplane)
            border_hyperplane = []
            for i in range(len(limits)):
                border_hyperplane.append(0)
            border_hyperplane.append(-1)
            border_hyperplane.append(2*g_max)
            # print(border_hyperplane)
            border_hyperplanes.append(border_hyperplane)
            
            # print("border_hyperplanes",border_hyperplanes)
            
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
#             CP_volume = {}
            for entry, points in CPdomains.items():
                print(entry.name)
#                 points: PH,(other els),E,(1/R),PHI len(limits)+1
                if show_phi:
                    points = np.array(points)[:,:len(limits)+1] #+1 is for phi values
                else:
                    points = np.array(points)[:,:len(limits)]
                # Initial sort to ensure consistency
                points = points[np.lexsort(np.transpose(points))]
                center = np.average(points, axis=0)
                points_centered = points - center 
                point_comparator = lambda x, y: x[0] * y[1] - x[1] * y[0]
                points_centered = sorted(points_centered,
                                         key=cmp_to_key(point_comparator))
                points = points_centered + center
                if len(points) < 4:
                    points = np.vstack((points, points[0]))
                # print(points)
                hull = ConvexHull(points,qhull_options='QJ')
#                     CP_volume[entry] = hull.volume
                # for indices in hull.simplices:
                #     print(indices)
                #     print(points[indices])
                simplices = [points[indices] for indices in hull.simplices]

                CPdomains[entry] = simplices
                CP_domain_vertices[entry] = points
                print(points)



        return CPdomains, CP_domain_vertices
    
    
    def get_phase_coexistence_region_w_aqueous_constraint(self):
        '''
        return a phase coexistence dict,
        key: "phase2","phase3","phase4","phase5"
        value: list, the first element is the list of names
                     the second element is the list of shared vertices
        '''
        stable_domain_vertices = self._stable_domain_vertices1
        
        phaseCoex = {"phase2":[],"phase3":[],"phase4":[],"phase5":[]}
        '''2-phase coexistence'''
        for entry1, entry2 in list(combinations(stable_domain_vertices.keys(),2)):
            # print(entry1.name,entry2.name)
            # if sorted([entry1.name, entry2.name]) != sorted(["Mn^{2+}", "beta-MnO2"]):
            #     continue
            print(entry1.name)
            stable_domain_vertices_entry1 = np.around(
                stable_domain_vertices[entry1], decimals = 6).tolist()
            stable_domain_vertices_entry2 = np.around(
                stable_domain_vertices[entry2], decimals = 6).tolist()
            # stable_domain_vertices_entry1 = stable_domain_vertices[entry1].tolist()
            # stable_domain_vertices_entry2 = stable_domain_vertices[entry2].tolist()
            sharedvers=[]

            for ver in stable_domain_vertices_entry1:
                # print(stable_domain_vertices[entry1])
                
                if ver in stable_domain_vertices_entry2:
                    if ver not in sharedvers:
                        sharedvers.append(ver)
                    # if np.around(ver, decimals=10).tolist() not in sharedvers:
                    #     sharedvers.append(np.around(ver, decimals=10).tolist())

            namelist = [i[0] for i in phaseCoex["phase2"]]
            names = [entry1.name,entry2.name]
            names.sort()
            if len(sharedvers) >= 4 and names not in namelist:
                phaseCoex["phase2"].append([names,sharedvers])
        '''3-phase and 4-phase coexistence'''
        for phase2p1, phase2p2 in list(combinations(phaseCoex["phase2"],2)):
            sharedvers=[]
            for ver in phase2p1[1]:
                if ver in phase2p2[1]:
                    if ver not in sharedvers:
                        sharedvers.append(ver)
                    # ver = np.around(ver, decimals=10).tolist()
                    # if ver not in sharedvers:
                    #     sharedvers.append(ver)
            if len(sharedvers):
                names1 = phase2p1[0]
                names2 = phase2p2[0]
                names = []
                for i in names1+names2:
                    if i not in names:
                        names.append(i)
                names.sort()
                namelist3 = [i[0] for i in phaseCoex["phase3"]]
                namelist4 = [i[0] for i in phaseCoex["phase4"]]
                # if names not in list(phaseCoex["phase3"].keys()):
                if len(names)==3 and len(sharedvers) >= 3 and names not in namelist3:
                    phaseCoex["phase3"].append([names, sharedvers])
                if len(names)==4 and len(sharedvers) >= 2 and names not in namelist4:
                    phaseCoex["phase4"].append([names, sharedvers])
        '''5-phase coexistence'''
        for phase4p1, phase4p2 in list(combinations(phaseCoex["phase4"],2)):
            sharedvers=[]
            for ver in phase4p1[1]:
                if ver in phase4p2[1]:
                    sharedvers.append(ver)
            namelist5 = [i[0] for i in phaseCoex["phase5"]]
            if len(sharedvers):
                names1 = phase4p1[0]
                names2 = phase4p2[0]
                names = []
                for i in names1+names2:
                    if i not in names:
                        names.append(i)
                names.sort()
                if len(names)==5 and names not in namelist5:
                    phaseCoex["phase5"].append([names, sharedvers])

        return phaseCoex
        
    
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


class GeneralizedPlotter:
    """
    A plotter class for phase diagrams.

    Args:
        phasediagram: A PhaseDiagram object.
        show_unstable: Whether unstable phases will be plotted as well as
            red crosses. Defaults to False.
    """

    def __init__(self, generalizedDiagram, elementList = None, fixed = None):
        self._cp = generalizedDiagram
        
        if elementList == None:
            self.elementList = generalizedDiagram.elementList
        else:
            self.elementList = elementList
            
        if fixed == None:
            self.sliceTitle = False
        else:
            self.sliceTitle = True
        self.fixed = fixed
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
        
    def get_lnp_ammonia_pourbaix_diagram(
            self,limits = None,label_domains=True,
            bold_boundary=True,show_phi=False,
            edc = None, alpha = 0.3, vis_axis=["PH","K","E"]):

        stable_domain_vertices_constraints = self._cp._stable_domain_vertices1

        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]
        fix1 = [i for i in self.fixed if i[0] in self.elementList]
        fixed2 = [i for i in self.fixed if i[0] not in self.elementList]
        fixVar = {ee:val for ee,val in fixed2}
        if limits is None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
            if self._cp.surf:
                limits+=[[0,1]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != "N":
                    newlimits += [limits[i]]
            newlimits += limits[C:]
            limits = newlimits

        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c
            # print(e.name, c)
        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}

        print(limits)

        if len(vis_axis)==3:
            eremain = self.elementList.copy()
    
            for e in fix1:
                eremain.remove(e[0])
            eremain.remove("N")
    
            if "E" not in fixVar:
                eremain += ["E"]
            if "logP" not in fixVar:
                eremain += ["logP"]
            if self._cp.extra_comp_axis:
                for el in self._cp.extra_comp_axis:
                    if "log"+el not in fixVar:
                        eremain += ["log"+el]
    
            indexes = []
            for i in vis_axis:
                if i=="PH":
                    i="H"
                indexes.append(eremain.index(i))
            print(indexes)
            print(eremain)

            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[indexes[0]])
            ax.set_ylim(limits[indexes[1]])
            ax.set_zlim(limits[indexes[2]])
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                elif vis_axis.index("PH") == 1:
                    ax.set_ylim([0,14])
                else:
                    ax.set_zlim([0,14])
            plt.xlabel(vis_axis[0], **label_font)
            plt.ylabel(vis_axis[1], **label_font)
            ax.set_zlabel("logP", **label_font)
            namesubs = ["alpha","beta","gamma","delta"]
            namesvis = ['H^{1+}', 'Mn^{2+}', 'alpha-K0.11MnO1.94', 'beta-MnO2', 'gamma-MnOOH']
#             ax.scatter3D(0.1681955,0,1.135737)
            for entry in stable_domain_vertices_constraints:
#                 if entry.name != "GaN":
#                     continue
                org_vers = stable_domain_vertices_constraints[entry]
                vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])
                vertices = np.hstack([vertices,org_vers[:,[indexes[2]]]])
                print(entry.name)
                print(len(vertices))
                try:
                    hull = ConvexHull(vertices)
                except:
                    hull = ConvexHull(vertices,qhull_options = "QJ")
                simplices = hull.simplices 
                org_triangles = [vertices[s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, \
                     alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],entry.symbol,c=colordict[entry.name],ha='center',
                            va='center',**text_font,zorder = 40)
                if bold_boundary:
                    lines = []
                    for tri in org_triangles:
                        lines += list(combinations(tri.tolist(),2))
                    print(lines[0])
                    lines1 = []
                    for li in lines:
                        v1=li[0]
                        v2=li[1]
                        num_planes_line_shared = 0
                        print(v1)
                        print(v2)
                        eqs = hull.equations
                        eqs = np.unique(eqs, axis = 0)
                        for eq in eqs:
                            if abs(np.dot(eq[0:3],v1)+eq[3]) <1e-6 \
                                and abs(np.dot(eq[0:3],v2)+eq[3]) <1e-6:
                                    num_planes_line_shared += 1
                                    # print(abs(np.dot(eq[0:3],v1)+eq[3]),abs(np.dot(eq[0:3],v2)+eq[3]))
                                    # print()
                                    print(eq)
                        print(num_planes_line_shared)
                        if num_planes_line_shared == 2:
                            lines1.append(li)
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.5,linewidth=2)
                
        plt.show()  
              
    def get_magnetic_pourbaix_diagram(
            self,limits = None,label_domains=True,
            bold_boundary=True,show_phi=False,
            edc = None, alpha = 0.3, vis_axis=["PH","K","E"]):
        # We can visualize random 3 from 4 variables (PH mu_K E 1/R)
        
#         stable_domain_constraints, stable_domain_vertices_constraints=\
#             self._cp.get_generalized_chem_pot_domains_w_aqueous_constraints(
#                     self._cp._processed_entries, self._cp.elementList, 
#                     fixed=self.fixed, mutoPH=self._cp.mutoPH, 
#                     limits=limits,
#                     show_phi=show_phi)
        stable_domain_vertices_constraints = self._cp._stable_domain_vertices1
        eremain = self.elementList.copy()
        for e in self.fixed:
            eremain.remove(e[0])
        eremain.remove("O")
        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]
        
        if limits is None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
            if self._cp.surf:
                limits+=[[0,1]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != "O":
                    newlimits += [limits[i]]
            newlimits += limits[C:]
            limits = newlimits

        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c
            # print(e.name, c)
        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        print(eremain)
        print(limits)
        if len(vis_axis)==3:
            hindex = eremain.index("H")
            eremain +=["E","1/R"]
            newlimits = []
            indexes = []
            for i in vis_axis:
                if i=="PH":
                    i="H"
                indexes.append(eremain.index(i))
                # newlimits.append(limits[eremain.index(i)])
            # print(newlimits)    
            print(indexes)  
            
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[indexes[0]])
            ax.set_ylim(limits[indexes[1]])
            ax.set_zlim(limits[indexes[2]])
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                elif vis_axis.index("PH") == 1:
                    ax.set_ylim([0,14])
                else:
                    ax.set_zlim([0,14])
            plt.xlabel(vis_axis[0], **label_font)
            plt.ylabel(vis_axis[1], **label_font)
            ax.set_zlabel("H", **label_font)
            namesubs = ["alpha","beta","gamma","delta"]
            namesvis = ['H^{1+}', 'Mn^{2+}', 'alpha-K0.11MnO1.94', 'beta-MnO2', 'gamma-MnOOH']
#             ax.scatter3D(0.1681955,0,1.135737)
            for entry in stable_domain_vertices_constraints:

                org_vers = stable_domain_vertices_constraints[entry]
                vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])
                vertices = np.hstack([vertices,org_vers[:,[indexes[2]]]])
                print(entry.name)
                print(len(vertices))
                try:
                    hull = ConvexHull(vertices)
                except:
                    hull = ConvexHull(vertices,qhull_options = "QJ")
                simplices = hull.simplices 
                org_triangles = [vertices[s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, \
                     alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],entry.symbol,c=colordict[entry.name],ha='center',
                            va='center',**text_font,zorder = 40)
                if bold_boundary:
                    lines = []
                    for tri in org_triangles:
                        lines += list(combinations(tri.tolist(),2))
                    print(lines[0])
                    lines1 = []
                    for li in lines:
                        v1=li[0]
                        v2=li[1]
                        num_planes_line_shared = 0
                        print(v1)
                        print(v2)
                        eqs = hull.equations
                        eqs = np.unique(eqs, axis = 0)
                        for eq in eqs:
                            if abs(np.dot(eq[0:3],v1)+eq[3]) <1e-6 \
                                and abs(np.dot(eq[0:3],v2)+eq[3]) <1e-6:
                                    num_planes_line_shared += 1
                                    # print(abs(np.dot(eq[0:3],v1)+eq[3]),abs(np.dot(eq[0:3],v2)+eq[3]))
                                    # print()
                                    print(eq)
                        print(num_planes_line_shared)
                        if num_planes_line_shared == 2:
                            lines1.append(li)
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.5,linewidth=2)
                
        plt.show() 
    def get_high_dimensional_pourbaix_diagram(
            self,limits = None,label_domains=True,
            bold_boundary=True,show_phi=False,
            edc = None, alpha = 0.3, vis_axis=["PH","K","E"]):
        # We can visualize random 3 from 4 variables (PH mu_K E 1/R)
        
#         stable_domain_constraints, stable_domain_vertices_constraints=\
#             self._cp.get_generalized_chem_pot_domains_w_aqueous_constraints(
#                     self._cp._processed_entries, self._cp.elementList, 
#                     fixed=self.fixed, mutoPH=self._cp.mutoPH, 
#                     limits=limits,
#                     show_phi=show_phi)
        stable_domain_vertices_constraints = self._cp._stable_domain_vertices1
        eremain = self.elementList.copy()
        for e in self.fixed:
            eremain.remove(e[0])
        eremain.remove("O")
        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]
        
        if limits is None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
            if self._cp.surf:
                limits+=[[0,1]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != "O":
                    newlimits += [limits[i]]
            newlimits += limits[C:]
            limits = newlimits

        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c
            # print(e.name, c)
        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        print(eremain)
        print(limits)
        if len(vis_axis)==3:
            hindex = eremain.index("H")
            eremain +=["E","1/R"]
            newlimits = []
            indexes = []
            for i in vis_axis:
                if i=="PH":
                    i="H"
                indexes.append(eremain.index(i))
                # newlimits.append(limits[eremain.index(i)])
            # print(newlimits)    
            print(indexes)  
            
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[indexes[0]])
            ax.set_ylim(limits[indexes[1]])
            ax.set_zlim(limits[indexes[2]])
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                elif vis_axis.index("PH") == 1:
                    ax.set_ylim([0,14])
                else:
                    ax.set_zlim([0,14])
            plt.xlabel(vis_axis[0],**label_font)
            plt.ylabel(vis_axis[1],**label_font)
            ax.set_zlabel(vis_axis[2],**label_font)
            namesubs = ["alpha","beta","gamma","delta"]
            namesvis = ['H^{1+}', 'Mn^{2+}', 'alpha-K0.11MnO1.94', 'beta-MnO2', 'gamma-MnOOH']
#             ax.scatter3D(0.1681955,0,1.135737)
            for entry in stable_domain_vertices_constraints:

                org_vers = stable_domain_vertices_constraints[entry]
                vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])
                vertices = np.hstack([vertices,org_vers[:,[indexes[2]]]])
                print(entry.name)
                print(len(vertices))
                try:
                    hull = ConvexHull(vertices)
                except:
                    hull = ConvexHull(vertices,qhull_options = "QJ")
                simplices = hull.simplices 
                org_triangles = [vertices[s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, \
                     alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],entry.name,c=colordict[entry.name],ha='center',
                            va='center',**text_font,zorder = 40)
                if bold_boundary:
                    lines = []
                    for tri in org_triangles:
                        lines += list(combinations(tri.tolist(),2))
                    print(lines[0])
                    lines1 = []
                    for li in lines:
                        v1=li[0]
                        v2=li[1]
                        num_planes_line_shared = 0
                        print(v1)
                        print(v2)
                        eqs = hull.equations
                        eqs = np.unique(eqs, axis = 0)
                        for eq in eqs:
                            if abs(np.dot(eq[0:3],v1)+eq[3]) <1e-6 \
                                and abs(np.dot(eq[0:3],v2)+eq[3]) <1e-6:
                                    num_planes_line_shared += 1
                                    # print(abs(np.dot(eq[0:3],v1)+eq[3]),abs(np.dot(eq[0:3],v2)+eq[3]))
                                    # print()
                                    print(eq)
                        print(num_planes_line_shared)
                        if num_planes_line_shared == 2:
                            lines1.append(li)
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.5,linewidth=2)
                
        plt.show() 
#         return plt
    def get_generalized_pourbaix_slice(
            self,limits = None,label_domains=True,
            bold_boundary=True,
            edc = None, alpha = 0.3, 
            vis_axis=["PH","K","E"],
            vis_phase1 = False,
            vis_phase2 = False,
            fig = None,
            fcs = None):
        # We can visualize random 3 from 4 variables (PH mu_K E 1/R)
        stable_domain_vertices_constraints = self._cp._stable_domain_vertices1
        # for entry in stable_domain_vertices_constraints:
        #     print(entry.name)
        #     print(stable_domain_vertices_constraints[entry])
        #     gaga
        
        fix1 = [i for i in self.fixed if i[0] in self.elementList]
        fixed2 = [i for i in self.fixed if i[0] not in self.elementList]
        fixVar = {ee:val for ee,val in fixed2}

        eremain = self.elementList.copy()
        for e in fix1:
            eremain.remove(e[0])
        eremain.remove("O")
        F = len(fix1)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(fix1[i][0]) for i in range(F)]
        
        if limits is None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
            if self._cp.surf:
                limits+=[[0,1]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != "O":
                    newlimits += [limits[i]]
            if len(fixed2) == 0:
                newlimits += limits[C:]
            else:
                
                if "E" not in fixVar:
                    newlimits += [limits[-2]]
                if "1/R" not in fixVar:
                    newlimits += [limits[-1]]
            limits = newlimits
        
        '''add the phase coexistence you want to visualize here'''
        # the function below find shared vertices by different phases, leading to phase coexistence regions
        phaseCoex = self._cp.get_phase_coexistence_region_w_aqueous_constraint()
#         phas2 = [sorted(['beta-MnO2', 'delta-K0.21MnO1.87']), sorted(['alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87']), sorted(['alpha-K0.11MnO1.94', 'beta-MnO2'])]
#         phas2 = [sorted(['Mn^{2+}', 'delta-K0.21MnO1.87']), sorted(['alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87']), sorted(['alpha-K0.11MnO1.94', 'beta-MnO2'])]
        phas2 = [sorted(['Mn^{2+}', 'delta-K0.21MnO1.87']), sorted(['alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87']), sorted(['alpha-MnOOH', 'beta-MnO2'])]
        phas3 = [sorted(['alpha-K0.11MnO1.94', 'Mn^{2+}', 'delta-K0.21MnO1.87'])]
        phas4 = [sorted(['alpha-K0.11MnO1.94', 'beta-MnO2', 'delta-K0.21MnO1.87', "Mn^{2+}"])]
        phas5 = [sorted(['alpha-K0.11MnO1.94', 'beta-MnO2', 'delta-K0.21MnO1.87', "Mn^{2+}",'gamma-MnOOH'])]
#         phas3 = ['alpha-K0.11MnO1.94', 'alpha-Mn2O3','alpha-MnO2']
        phast = [sorted(phas2),sorted(phas3),sorted(phas4),sorted(phas5)]
        from collections import defaultdict
        newphaseCoex = defaultdict(list)
        for phas, nnn in zip(phast, list(phaseCoex.keys())):
            for names,ver in phaseCoex[nnn]:
                names.sort()
                if names in phas:
                    newphaseCoex[nnn].append([names, ver])
#                     print(names)

        phaseCoex = newphaseCoex
        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c
            # print(e.name, c)
        colordict = {'R-MnO2': [1.0, 0.0, 0.16, 1.0], 'alpha-K0.11MnO1.94': [1.0, 0.17700052994170645, 0.0, 1.0], 'alpha-Mn2O3': [1.0, 0.5161632220455751, 0.0, 1.0], 'beta-MnO2': [1.0, 0.8553259141494436, 0.0, 1.0], 'delta-K0.21MnO1.87': [0.12718600953895065, 1.0, 0.0, 1.0], 'delta-K0.5MnO2': [0.4663487016428193, 1.0, 0.0, 1.0], 'delta-K0.75MnO2': [0.12718600953895065, 1.0, 0.0, 1.0], 'delta-MnO2': [0.0, 1.0, 0.2108370229812356, 1.0], 'gamma-MnOOH': [0.0, 1.0, 0.5481762597512125, 1.0], 'MnO': [0.0, 1.0, 0.8855154965211895, 1.0], 'Mn3O4': [0.0, 0.774722932651321, 1.0, 1.0], 'KMnO2': [0.0, 0.43371696504688806, 1.0, 1.0], 'Mn^{2+}': [0.0, 0.09271099744245515, 1.0, 1.0], 'MnO4^{1-}': [0.24829497016197802, 0.0, 1.0, 1.0], 'alpha-MnOOH': [0.5893009377664112, 0.0, 1.0, 1.0], 'Mn^{3+}': [0.9303069053708444, 0.0, 1.0, 1.0], 'MnO4^{2-}': [1.0, 0.0, 0.75, 1.0]}

        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        print("eremain",eremain)
        print("limits",limits)
        
        if len(vis_axis)==2:
            if "E" not in fixVar:
                eremain += ["E"]
            if "1/R" not in fixVar:
                eremain += ["1/R"]
            print(eremain)
            indexes = []
            for i in vis_axis:
                if i=="PH":
                    i="H"
                indexes.append(eremain.index(i))
            print(indexes)



            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()

            ax.set_xlim(limits[indexes[0]])

            ax.set_ylim(limits[indexes[1]])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                 grid_color='k', grid_alpha=0.5)
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                else:
                    ax.set_ylim([0,14])

            plt.xlabel(vis_axis[0],**label_font)
            plt.ylabel(vis_axis[1],**label_font)
            plt.rcParams['text.usetex'] = True
            if vis_phase1:
#                 phase1s = ['Mn3O4', 'Mn^{2+}', 'alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87', 'gamma-MnOOH']
                phase1s = ['gamma-MnOOH', 'Mn^{2+}', "Mn3O4","alpha-Mn2O3"]
                phase1s = ['beta-MnO2',"R-MnO2","delta-MnO2","Mn^{2+}"]
                for entry, sharedvers in stable_domain_vertices_constraints.items():
#                     if entry.name not in phase1s:
#                         continue
                    if "K" in entry.name:
                        continue
                    sharedvers = np.array(sharedvers)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])

                    fc = colordict[entry.name]
                    if fcs:
                        fc = "pink"
                        if entry.name == "Mn3O4":
                            fc = "blue"

                    new_vertices = np.array(vertices)
                    print("new_vertices",new_vertices)

                    if len(new_vertices) == 2: 
                        '''projection in this space is a line'''
                        ax.plot(new_vertices[:,0],new_vertices[:,1],color = fc,alpha = alpha,linewidth = 1)
                        continue
                    if len(new_vertices) == 1:
                        ax.scatter(new_vertices[:,0],new_vertices[:,1],s = 60,color = fc,alpha = alpha,linewidth = 1)
                        continue
                    hull = ConvexHull(new_vertices)
                    simplices = hull.simplices 
                    edges = [new_vertices[s] for s in simplices]
                    
                    plt.fill(new_vertices[hull.vertices][:,0], new_vertices[hull.vertices][:,1], 
                             alpha = 0.3,facecolor = fc)
                    if label_domains:
                        center = np.average(vertices, axis=0)

                        ax.text(center[0],center[1], entry.symbol,c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if bold_boundary:
                        for li in edges:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],color='k',alpha = 0.6,linewidth=1)
#             ax.invert_yaxis()
        if len(vis_axis)==3:
            hindex = eremain.index("H")

            newlimits = []
            if "E" not in fixVar:
                eremain += ["E"]
            if "1/R" not in fixVar:
                eremain += ["1/R"]
            print(eremain)
            indexes = []
            for i in vis_axis:
                if i=="PH":
                    i="H"
                indexes.append(eremain.index(i))
            print(indexes)

            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=166
            ax.elev=38
            ax.set_xlim(limits[indexes[0]])
            ax.set_ylim(limits[indexes[1]])
            ax.set_zlim(limits[indexes[2]])
            
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                elif vis_axis.index("PH") == 1:
                    ax.set_ylim([0,14])
                else:
                    ax.set_zlim([0,14])
            plt.xlabel(vis_axis[0],**label_font)
            plt.ylabel(vis_axis[1],**label_font)
            # ax.set_yticks([0,0.5,1,1.5,2])
            ax.set_zlabel(vis_axis[2],**label_font)
            fn=0 

            if vis_phase1:
                phases1 = ['Mn3O4', 'Mn^{2+}', 'alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87', 'gamma-MnOOH']
                phases1 = ['Mn^{2+}',"alpha-K0.11MnO1.94",'Mn3O4','gamma-MnOOH',"alpha-MnO2"]
                phases1 = ['Mn^{2+}',"alpha-K0.11MnO1.94",'delta-K0.21MnO1.87',"beta-MnO2","alpha-MnOOH"]
                phasedc = {"Mn3O4":"blue", "alpha-MnO2":"purple", 'alpha-Mn2O3':"gray", 'Mn^{2+}':"yellow",'alpha-K0.11MnO1.94':"cyan",'delta-K0.21MnO1.87':"green",'gamma-MnOOH':"pink"}
                # phases1 = ["Mn3O4"]
                for entry in stable_domain_vertices_constraints:
                    
                    # if entry.name.split("-")[0] != "alpha":
                    #     continue
                    if entry.name not in phases1:
                        continue
    #                 if entry.name not in namesvis:
    #                     continue
                    org_vers = stable_domain_vertices_constraints[entry]

                    vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])
                    vertices = np.hstack([vertices,org_vers[:,[indexes[2]]]])

                    try:
                        hull = ConvexHull(vertices)
                    except:
                        hull = ConvexHull(vertices,qhull_options = "QJ")
                    simplices = hull.simplices 
                    org_triangles = [vertices[s] for s in simplices]

                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                    ax.add_collection3d(pc)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        if entry.name == "beta-MnO2":
                            center[-1] *= 0.5
                        if entry.name == "delta-K0.21MnO1.87":
                            center[-1] *= 3.5
                            center[0] *= 2
                        if entry.name == "Mn^{2+}":
                            center[-1] *= 1.8
                            center[0] *= 0.8
                        ax.text(center[0],center[1],center[2],entry.symbol,c=colordict[entry.name],ha='center',
                                va='center',**text_font,zorder = 40)

                    if bold_boundary:
                        lines1 = get_edges_of_polytope_in3D(org_triangles, hull)
 
                        for li in lines1:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.6,linewidth=0.5)
            if vis_phase2:
                for names, sharedvers in phaseCoex["phase2"]:
#                     if names not in all_not_3d_names:
#                         continue
                    sharedvers = np.array(sharedvers)
                    # print(names)
                    # print(len(sharedvers))
                    # print(sharedvers)
                    # print(indexes)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])
                    vertices = np.hstack([vertices,sharedvers[:,[indexes[2]]]])

                    fc = np.average([colordict[name] for name in names],axis=0)

                    new_vertices = []
                    for i in vertices.tolist():
                        '''around data can not be used for finding boundaries of 3d Polytopes,
                        we need original vertices to plot boundaries of 3d polytopes;
                        in the meanwhile, non-around data not work for 2d boundaries of polygons'''
                        i=np.around(i, decimals=10).tolist()
                        if i not in new_vertices:
                            new_vertices.append(i)
                    new_vertices = np.array(new_vertices)
                    print("new_vertices",new_vertices)

                    if len(new_vertices.tolist()) <= 2: 
                        '''projection in this space is a line'''

                        ax.plot(new_vertices[:,0],new_vertices[:,1],new_vertices[:,2],color = fc,alpha = alpha,linewidth = 1)
                        continue
                    
                    vertices_add = np.copy(vertices) 
                    '''works for boundries of 3d polytopes, has to use original vertices'''
                    while len(vertices_add.tolist()) < 4:
                        '''vertices number is < 4, convex hull algorithm can not work for 3d'''
                        vertices_add = np.vstack((vertices_add,vertices_add[0]))
                    hull_3d = True
                    print(names)

                    try:
                        '''if bold boundaries of 3d polytopes, hull can not be QJ option''' 
                        hull = ConvexHull(vertices_add)
                    except:
                        '''always, 2d polygons (coplanar) can not be calculated in 3d.'''
                        hull_3d = False
#                         all_not_3d_names.append(names)
                        hull = ConvexHull(vertices_add,qhull_options = "QJ")
                    print(names, hull_3d)

                    simplices = hull.simplices 
                    org_triangles = [vertices_add[s] for s in simplices]
                    if fcs:
                        fc = fcs[0]
                    if names == sorted(["Mn^{2+}", "gamma-MnOOH"]):
                        fc = "green"
                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = alpha, facecolor=fc,edgecolor = edc,zorder=100)
                    ax.add_collection3d(pc)
                    if not label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if bold_boundary:

                        lines1 = get_edges_of_plane_in3D(new_vertices)
                        # if len(lines1) < 3:
                        #     lines1 = get_edges_of_plane_in3D(new_vertices)
                        # print(lines1)

                        for li in lines1:

                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.6,linewidth=2)

          
            if vis_axis[2] == "1/R":
                ax.invert_zaxis()
            ax.text(4, -2, -0.75, "E = " + str(round(fixVar["E"],3)), **label_font)  
        # ax.set_axis_off()
        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        a = " ".join([e + "_"+ str(fixVar[e]) for e in fixVar])
        
        ax.set_title(a)

        return ax
#         plt.show() 

    def get_slice_projection_schematic_figure(
            self,limits = None,label_domains=True,
            bold_boundary=True,
            edc = None, alpha = 0.3, 
            vis_axis=["PH","K","E"],
            fig = None,
            example = ["alpha-MnOOH"]):
        # We can visualize random 3 from 4 variables (PH mu_K E 1/R)
        stable_domain_vertices_constraints = self._cp._stable_domain_vertices1

        eremain = self.elementList.copy()
        for e in self.fixed:
            eremain.remove(e[0])
        eremain.remove("O")
        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]
        
        if limits is None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
            if self._cp.surf:
                limits+=[[0,1]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != "O":
                    newlimits += [limits[i]]
            newlimits += limits[C:]
            limits = newlimits
        print(limits)

        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c.tolist()
            # print(e.name, c)

        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'10', 'weight':'normal'}
        print("eremain",eremain)
        print("limits",limits)

        if len(vis_axis)==3:
            hindex = eremain.index("H")
            eremain +=["E","1/R"]
            newlimits = []
            indexes = []
            for i in vis_axis:
                if i=="PH":
                    i="H"
                indexes.append(eremain.index(i))


            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[indexes[0]])
            ax.set_ylim(limits[indexes[1]])
            ax.set_zlim(limits[indexes[2]])
            
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                elif vis_axis.index("PH") == 1:
                    ax.set_ylim([0,14])
                else:
                    ax.set_zlim([0,14])
            plt.xlabel(vis_axis[0],**label_font)
            plt.ylabel(vis_axis[1],**label_font)
            ax.set_yticks([0,0.5,1,1.5,2])
            ax.set_zlabel(vis_axis[2],**label_font)
            
            # phases1 = ['Mn3O4',"gamma-MnOOH","alpha-MnOOH"]
            for entry in stable_domain_vertices_constraints:
                if entry.name not in example:
                    continue

                
                org_vers = stable_domain_vertices_constraints[entry]
                vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])
                vertices = np.hstack([vertices,org_vers[:,[indexes[2]]]])


                try:
                    hull = ConvexHull(vertices)
                except:
                    hull = ConvexHull(vertices,qhull_options = "QJ")
                simplices = hull.simplices 
                org_triangles = [vertices[s] for s in simplices]

                pc = a3.art3d.Poly3DCollection(org_triangles, \
                     alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],entry.name,c=colordict[entry.name],ha='center',
                            va='center',**text_font,zorder = 40)

                if bold_boundary:
                    lines1 = get_edges_of_polytope_in3D(org_triangles, hull)
            
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.6,linewidth=1.5)
                
                # plot projection
                vertices[:, -1] = 0.4
                for ver in vertices:
                    ax.scatter3D(ver[0], ver[1], ver[2], c = "blue")
                hull = ConvexHull(vertices,qhull_options = "QJ")
                simplices = hull.simplices 
                org_triangles = [vertices[s] for s in simplices]

                pc = a3.art3d.Poly3DCollection(org_triangles, \
                     alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if bold_boundary:
                    lines1 = get_edges_of_plane_in3D(vertices)
            
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color='blue',alpha = 0.6,linewidth=3)
                # GET SLICE
                fix_val = 0.25
#                 for fix_val in []
                fixed = [("Mn",0),["E",fix_val]]
        
                limits = [[-10,0],[-6,2],[-10,0],[-10,0]]
                limits += [[0,2],[0,1]]
                gpd = GeneralizedDiagram(self._cp._processed_entries,self._cp.elementList,
                                         fixed=fixed, limits=limits,
                                         surf=True,normalizedfactor="Nm",
                                         mutoPH=True,
                                         slicePlane=True,sliceLower=False,
                                         w_aqu_constraint=True)
                stable_domain_vertices_constraints_slice = gpd._stable_domain_vertices1
                for entry in stable_domain_vertices_constraints_slice:
                    if entry.name not in example:
                        continue
                    vertices = stable_domain_vertices_constraints_slice[entry]
                    # change first column value to fixed E(original data is for PH, K, 1/R)
                    for val in [fix_val, -0.6]:
                        vertices[:,1] = val
                        hull = ConvexHull(vertices,qhull_options = "QJ")
                        simplices = hull.simplices 
                        org_triangles = [vertices[s] for s in simplices]
        
                        pc = a3.art3d.Poly3DCollection(org_triangles, \
                             alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                        ax.add_collection3d(pc)
                        if bold_boundary:
                            lines1 = get_edges_of_plane_in3D(vertices)
                    
                            for li in lines1:
                                li = np.array(li)
                                ax.plot(li[:,0],li[:,1],li[:,2],color='red',alpha = 0.6,linewidth=3)
                
            # if vis_axis[2] == "1/R":
            #     ax.invert_zaxis()
        return ax
                # plt.show()
    def get_phase_coexistence(
            self,limits = None,label_domains=True,
            bold_boundary=True,
            edc = None, alpha = 0.3, 
            vis_axis=["PH","K","E"],
            vis_phase1 = False,
            vis_phase2 = False,
            vis_phase3 = False,
            vis_phase4 = False,
            vis_phase5 = False,
            fig = None,
            fcs = None):
        # We can visualize random 3 from 4 variables (PH mu_K E 1/R)
        stable_domain_vertices_constraints = self._cp._stable_domain_vertices1
        phaseCoex = self._cp.get_phase_coexistence_region_w_aqueous_constraint()
        for i in phaseCoex:
            print(i)
            for j in phaseCoex[i]:
                print(sorted(j[0]))
            print()
            
#         phas2 = ['delta-K0.21MnO1.87', 'gamma-MnOOH']
#         phas2 = ['Mn^{2+}', 'gamma-MnOOH']
#         phas3 = ['delta-K0.21MnO1.87', 'gamma-MnOOH','Mn3O4']
#         phas4 = ['alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87', 'gamma-MnOOH','Mn3O4']
#         phas5 = ['Mn3O4', 'Mn^{2+}', 'alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87', 'gamma-MnOOH']
# #         phas3 = ['alpha-K0.11MnO1.94', 'alpha-Mn2O3','alpha-MnO2']
#         phas = [sorted(phas2),sorted(phas3),sorted(phas4),sorted(phas5)]
# #         for names,ver in phaseCoex["phase5"]:
# #             print(names)
#         for phass,nnn in zip(phas,list(phaseCoex.keys())):
#             for names,ver in phaseCoex[nnn]:
# #                 print(names, ver)
#                 if phass == names:
# #                     print(names)
# #                     print(ver)
#                     phaseCoex[nnn] = [[names,ver]]
#                     break


        phas2 = [sorted(['beta-MnO2', 'delta-K0.21MnO1.87']), sorted(['alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87']), sorted(['alpha-K0.11MnO1.94', 'beta-MnO2'])]
        phas2 = [sorted(['alpha-MnOOH', 'Mn^{2+}'])]
        phas3 = [sorted(['alpha-K0.11MnO1.94', 'Mn^{2+}', 'delta-K0.21MnO1.87'])]
        phas4 = [sorted(['alpha-K0.11MnO1.94', 'beta-MnO2', 'delta-K0.21MnO1.87', "Mn^{2+}"])]
        phas5 = [sorted(['Mn^{2+}', 'alpha-K0.11MnO1.94', 'alpha-MnOOH', 'beta-MnO2', 'delta-K0.21MnO1.87'])]

#         phas3 = ['alpha-K0.11MnO1.94', 'alpha-Mn2O3','alpha-MnO2']
        phast = [sorted(phas2),sorted(phas3),sorted(phas4),sorted(phas5)]
        from collections import defaultdict
        newphaseCoex = defaultdict(list)
        for phas, nnn in zip(phast, list(phaseCoex.keys())):
            for names,ver in phaseCoex[nnn]:
                names.sort()
                if names in phas:
                    newphaseCoex[nnn].append([names, ver])
#                     print(names)

        phaseCoex = newphaseCoex
        for i in phaseCoex["phase3"]:
            print(i[0], len(i[1]))
            print(np.array(i[1]))

#         gaga

        eremain = self.elementList.copy()
        for e in self.fixed:
            eremain.remove(e[0])
        eremain.remove("O")
        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]
        
        if limits is None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
            if self._cp.surf:
                limits+=[[0,1]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != "O":
                    newlimits += [limits[i]]
            newlimits += limits[C:]
            limits = newlimits
        print(limits)

        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c.tolist()
            # print(e.name, c)
        colordict = {'R-MnO2': [1.0, 0.0, 0.16, 1.0], 'alpha-K0.11MnO1.94': [1.0, 0.17700052994170645, 0.0, 1.0], 'alpha-Mn2O3': [1.0, 0.5161632220455751, 0.0, 1.0], 'beta-MnO2': [1.0, 0.8553259141494436, 0.0, 1.0], 'delta-K0.21MnO1.87': [0.12718600953895065, 1.0, 0.0, 1.0], 'delta-K0.5MnO2': [0.4663487016428193, 1.0, 0.0, 1.0], 'delta-K0.75MnO2': [0.12718600953895065, 1.0, 0.0, 1.0], 'delta-MnO2': [0.0, 1.0, 0.2108370229812356, 1.0], 'gamma-MnOOH': [0.0, 1.0, 0.5481762597512125, 1.0], 'MnO': [0.0, 1.0, 0.8855154965211895, 1.0], 'Mn3O4': [0.0, 0.774722932651321, 1.0, 1.0], 'KMnO2': [0.0, 0.43371696504688806, 1.0, 1.0], 'Mn^{2+}': [0.0, 0.09271099744245515, 1.0, 1.0], 'MnO4^{1-}': [0.24829497016197802, 0.0, 1.0, 1.0], 'alpha-MnOOH': [0.5893009377664112, 0.0, 1.0, 1.0], 'Mn^{3+}': [0.9303069053708444, 0.0, 1.0, 1.0], 'MnO4^{2-}': [1.0, 0.0, 0.75, 1.0]}

        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        print("eremain",eremain)
        print("limits",limits)
        
        if len(vis_axis)==2:
            eremain +=["E","1/R"]
            newlimits = []
            indexes = []
            for i in vis_axis:
                if i=="PH":
                    i="H"
                indexes.append(eremain.index(i))
            # print(indexes)

            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()

            ax.set_xlim(limits[indexes[0]])
            ax.set_ylim(limits[indexes[1]])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                 grid_color='k', grid_alpha=0.5)
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                else:
                    ax.set_ylim([0,14])

            plt.xlabel(vis_axis[0],**label_font)
            plt.ylabel(vis_axis[1],**label_font)
            
                
            fn=0 
            if vis_phase1:
#                 phase1s = ['Mn3O4', 'Mn^{2+}', 'alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87', 'gamma-MnOOH']
                
                phase1s = ['gamma-MnOOH', 'Mn^{2+}', "Mn2O3"]
                # all_not_3d_names = [['H^{1+}', 'alpha-K0.11MnO1.94'], ['alpha-Mn2O3', 'alpha-MnO2'], ['alpha-Mn2O3', 'gamma-MnOOH'], ['Mn3O4', 'alpha-Mn2O3'], ['H^{1+}', 'alpha-Mn2O3'], ['alpha-MnO2', 'beta-MnO2'], ['alpha-MnO2', 'gamma-MnOOH'], ['H^{1+}', 'alpha-MnO2'], ['MnO4^{1-}', 'alpha-MnO2'], ['beta-MnO2', 'gamma-MnOOH'], ['H^{1+}', 'beta-MnO2'], ['Mn^{2+}', 'beta-MnO2'], ['MnO4^{1-}', 'beta-MnO2'], ['Mn3O4', 'gamma-MnOOH'], ['H^{1+}', 'gamma-MnOOH'], ['Mn^{2+}', 'gamma-MnOOH'], ['Mn(OH)2', 'Mn3O4'], ['H^{1+}', 'Mn3O4'], ['Mn3O4', 'Mn^{2+}'], ['H^{1+}', 'Mn(OH)2'], ['Mn(OH)2', 'Mn^{2+}'], ['H^{1+}', 'Mn^{2+}'], ['H^{1+}', 'MnO4^{1-}'], ['H^{1+}', 'K^{1+}']]
                for entry, sharedvers in stable_domain_vertices_constraints.items():
#                     if entry.name not in phase1s:
#                         continue
                    if "K" in entry.name:
                        continue

                    sharedvers = np.array(sharedvers)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])

                    fc = colordict[entry.name]
                    if fcs:
                        fc = "pink"
                        if entry.name == "Mn3O4":
                            fc = "blue"

                    new_vertices = np.array(vertices)
                    print("new_vertices",new_vertices)

                    if len(new_vertices) == 2: 
                        '''projection in this space is a line'''
                        ax.plot(new_vertices[:,0],new_vertices[:,1],color = fc,alpha = alpha,linewidth = 1)
                        continue
                    if len(new_vertices) == 1:
                        ax.scatter(new_vertices[:,0],new_vertices[:,1],s = 60,color = fc,alpha = alpha,linewidth = 1)
                        continue
                    hull = ConvexHull(new_vertices)
                    simplices = hull.simplices 
                    edges = [new_vertices[s] for s in simplices]
                    
                    plt.fill(new_vertices[hull.vertices][:,0], new_vertices[hull.vertices][:,1], 
                             alpha = 0.3,facecolor = fc)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],entry.name,c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if bold_boundary:
                        for li in edges:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],color='k',alpha = 0.6,linewidth=1)
            if vis_phase2:
                # all_not_3d_names = [['H^{1+}', 'alpha-K0.11MnO1.94'], ['alpha-Mn2O3', 'alpha-MnO2'], ['alpha-Mn2O3', 'gamma-MnOOH'], ['Mn3O4', 'alpha-Mn2O3'], ['H^{1+}', 'alpha-Mn2O3'], ['alpha-MnO2', 'beta-MnO2'], ['alpha-MnO2', 'gamma-MnOOH'], ['H^{1+}', 'alpha-MnO2'], ['MnO4^{1-}', 'alpha-MnO2'], ['beta-MnO2', 'gamma-MnOOH'], ['H^{1+}', 'beta-MnO2'], ['Mn^{2+}', 'beta-MnO2'], ['MnO4^{1-}', 'beta-MnO2'], ['Mn3O4', 'gamma-MnOOH'], ['H^{1+}', 'gamma-MnOOH'], ['Mn^{2+}', 'gamma-MnOOH'], ['Mn(OH)2', 'Mn3O4'], ['H^{1+}', 'Mn3O4'], ['Mn3O4', 'Mn^{2+}'], ['H^{1+}', 'Mn(OH)2'], ['Mn(OH)2', 'Mn^{2+}'], ['H^{1+}', 'Mn^{2+}'], ['H^{1+}', 'MnO4^{1-}'], ['H^{1+}', 'K^{1+}']]
                for names, sharedvers in phaseCoex["phase2"]:
                    # if names not in all_not_3d_names:
                    #     continue
                    sharedvers = np.array(sharedvers)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])

                    fc = np.average([colordict[name] for name in names],axis=0)
                    if fcs:
                        fc = fcs[0]
                    print()
                    print(names)
                    new_vertices = np.array(vertices)
                    # print("new_vertices",new_vertices)
                    if len(new_vertices) == 2: 
                        '''projection in this space is a line'''
                        ax.plot(new_vertices[:,0],new_vertices[:,1],color = fc,alpha = alpha,linewidth = 1)
                        continue
                    if len(new_vertices) == 1:
                        ax.scatter(new_vertices[:,0],new_vertices[:,1],s = 60,color = fc,alpha = alpha,linewidth = 1)
                        continue
                    hull = ConvexHull(new_vertices)
                    simplices = hull.simplices 
                    edges = [new_vertices[s] for s in simplices]
                    
                    plt.fill(new_vertices[hull.vertices][:,0], new_vertices[hull.vertices][:,1], 
                             alpha = 0.3,facecolor = fc)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if bold_boundary:
                        for li in edges:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],color='k',alpha = 0.6,linewidth=1)
                    # fn=fn+1
                    # print(names)
                    # fig.savefig("C:/Users/jdche/eclipse-workspace/research/myResearch/A220210_pourbaix_surface_energy/figs_phase2/" +"3d_"+str(fn) )
            if vis_phase3:
                for names, sharedvers in phaseCoex["phase3"]:
                    # if names not in all_not_3d_names:
                    #     continue
                    sharedvers = np.array(sharedvers)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])

                    fc = np.average([colordict[name] for name in names],axis=0)
                    if fcs:
                        fc = fcs[1]
                    print()
                    print(names)
                    new_vertices = np.array(vertices)
                    print("new_vertices",new_vertices)
                    # ax.plot(new_vertices[:,0],new_vertices[:,1],color = fc,alpha = 0.6,linewidth = 5)                    
                    if len(new_vertices) == 2: 
                        '''projection in this space is a line'''
                        ax.plot(new_vertices[:,0],new_vertices[:,1],color = fc,alpha = 1,linewidth = 3,
                                zorder = 100)
                        continue
                    if len(new_vertices) == 1:
                        ax.scatter(new_vertices[:,0],new_vertices[:,1],s = 60,color = fc)
                        continue
                    try:
                        hull = ConvexHull(new_vertices)
                    except:

                        ax.plot(new_vertices[:,0],new_vertices[:,1],color = fc,alpha = 0.6,linewidth = 5)

                        continue
                    
                    simplices = hull.simplices 
                    edges = [new_vertices[s] for s in simplices]
                    plt.fill(new_vertices[hull.vertices][:,0], new_vertices[hull.vertices][:,1], 
                             alpha = 0.3,facecolor = fc)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if bold_boundary:
                        for li in edges:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],color='k',alpha = 0.6,linewidth=1)
                    # fn=fn+1
                    # print(names)
                    # fig.savefig("C:/Users/jdche/eclipse-workspace/research/myResearch/" + 
                    #             "A220210_pourbaix_surface_energy/figs_phase3/" +"label_2d_"+str(fn) )

            if vis_phase4:
                for names, sharedvers in phaseCoex["phase4"]:
                    print()
                    print(names)
                    sharedvers = np.array(sharedvers)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])
                    print(len(vertices))

                    fc = np.average([colordict[name] for name in names],axis=0)
                    if fcs:
                        fc = fcs[2]
                    if len(vertices) == 2: 
                        ax.plot(vertices[:,0],vertices[:,1], color = fc,alpha = 1,linewidth = 5,
                                zorder = 100)
                    elif len(vertices) == 1:
                        ax.scatter(vertices[:,0],vertices[:,1], color = fc,alpha = 1)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    # fn=fn+1
                    # print(names)
                    # fig.savefig("C:/Users/jdche/eclipse-workspace/research/myResearch/" + 
                    #             "A220210_pourbaix_surface_energy/figs_phase4/" +"label_0d_"+str(fn) )
            if vis_phase5:
                for names, sharedvers in phaseCoex["phase5"]:
                    sharedvers = np.array(sharedvers)

                    fc = np.average([colordict[name] for name in names],axis=0)
                    if fcs:
                        fc = fcs[3]
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])

                    ax.scatter(vertices[:,0],vertices[:,1], s = 300,color = fc,zorder = 200)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 200)
            if vis_axis[-1] == "1/R":
                ax.invert_yaxis()
        if len(vis_axis)==3:
            hindex = eremain.index("H")
            eremain +=["E","1/R"]
            newlimits = []
            indexes = []
            for i in vis_axis:
                if i=="PH":
                    i="H"
                indexes.append(eremain.index(i))
            fn = 0
            print(indexes)
            
            # for zim in np.linspace(20, 88, 200):
            #     print(zim)
            #     fn += 1
            # #     if fn == 350:
            # #         break

            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=-90
            ax.elev=16

            ax.set_xlim(limits[indexes[0]])
            ax.set_ylim(limits[indexes[1]])
            ax.set_zlim(limits[indexes[2]])
            
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                elif vis_axis.index("PH") == 1:
                    ax.set_ylim([0,14])
                else:
                    ax.set_zlim([0,14])
            plt.xlabel(vis_axis[0],**label_font)
            plt.ylabel(vis_axis[1],**label_font)
#             ax.set_yticks([0,0.5,1,1.5,2])
            # ax.set_yticks([-9,-8,-7,-6,-5])
            ax.set_zlabel(vis_axis[2],**label_font)

            if vis_phase1:
                phases1 = ['Mn^{2+}',"delta-MnO2", 'delta-K0.21MnO1.87',"beta-MnO2",'alpha-K0.11MnO1.94']
                phases1 = ['KMnO2','delta-K0.21MnO1.87',"beta-MnO2",'alpha-K0.11MnO1.94']
                phases1 = ['delta-K0.21MnO1.87',"Mn^{2+}","beta-MnO2","alpha-K0.11MnO1.94"]
                # phases1 = ["MnO4^{1-}", "MnO4^{2-}","Mn^{2+}"]
                # phases1 = ["alpha-MnOOH","Mn^{2+}"]
                phasedc = {"Mn3O4":"blue", "alpha-MnO2":"purple", 'alpha-Mn2O3':"gray", 'Mn^{2+}':"yellow",'alpha-K0.11MnO1.94':"cyan",'delta-K0.21MnO1.87':"green",'gamma-MnOOH':"pink"}
                for entry in stable_domain_vertices_constraints:

                    # if "K" in entry.name:
                    #     continue
                    if vis_phase1 != 1:
                        if entry.name not in phases1:
                            continue                        
                    if entry.name == "alpha-MnOOH":
                        entry.symbol = r'$\%s$' % "alpha" + "-MnOOH"
                    org_vers = stable_domain_vertices_constraints[entry]
                    vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])

                    vertices = np.hstack([vertices,org_vers[:,[indexes[2]]]])

                    try:
                        hull = ConvexHull(vertices)
                    except:
                        hull = ConvexHull(vertices,qhull_options = "QJ")
                    simplices = hull.simplices 
                    org_triangles = [vertices[s] for s in simplices]
#                     if entry.name == "gamma-MnOOH":
#                         aa = 0.2
#                     elif entry.name == "Mn^{2+}":
#                         aa = 0.08
#                     elif entry.name == "Mn3O4":
#                         aa = 0.03
#                     else:
#                         aa = 0.05
                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = alpha, facecolor=colordict[entry.name],edgecolor = edc)
                    ax.add_collection3d(pc)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        if entry.name == "Mn^{2+}":
                            center[0] -= 2
                            center[-1] += 0.15
                            center[1] -= 0.25
                        if entry.name == "delta-K0.21MnO1.87":
                            center[0] += 2
                            center[-1] += 1
                            center[1] -= 0.1 
                        ax.text(center[0],center[1],center[2],entry.symbol,c=colordict[entry.name],ha='center',
                                va='center',**text_font,zorder = 40)
#                     if entry.name == "gamma-MnOOH":
#                         bold_boundary = True
#                     else:
#                         bold_boundary = False
                    if bold_boundary:
                        lines1 = get_edges_of_polytope_in3D(org_triangles, hull)
 
                        for li in lines1:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.6,
                                    linewidth=0.5)

            
            if vis_phase2:
                all_not_3d_names = [['alpha-Mn2O3', 'alpha-MnO2'], ['alpha-Mn2O3', 'gamma-MnOOH'], ['Mn3O4', 'alpha-Mn2O3'], ['alpha-MnO2', 'beta-MnO2'], ['alpha-MnO2', 'gamma-MnOOH'], ['MnO4^{1-}', 'alpha-MnO2'], ['beta-MnO2', 'gamma-MnOOH'], ['Mn3O4', 'gamma-MnOOH'], ['Mn^{2+}', 'gamma-MnOOH'], ['Mn(OH)2', 'Mn3O4'], ['Mn3O4', 'Mn^{2+}'], ['Mn(OH)2', 'Mn^{2+}']]
#                 all_not_3d_names = []


                for names, sharedvers in phaseCoex["phase2"]:
#                     if names not in all_not_3d_names:
#                         continue
                    sharedvers = np.array(sharedvers)
                    # print(names)
                    # print(len(sharedvers))

                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])
                    vertices = np.hstack([vertices,sharedvers[:,[indexes[2]]]])

                    fc = np.average([colordict[name] for name in names],axis=0)

                    new_vertices = []
                    for i in vertices.tolist():
                        '''around data can not be used for finding boundaries of 3d Polytopes,
                        we need original vertices to plot boundaries of 3d polytopes;
                        in the meanwhile, non-around data not work for 2d boundaries of polygons'''
                        i=np.around(i, decimals=10).tolist()
                        if i not in new_vertices:
                            new_vertices.append(i)
                    new_vertices = np.array(new_vertices)
                    print("new_vertices",new_vertices)

                    if len(new_vertices.tolist()) <= 2: 
                        '''projection in this space is a line'''

                        ax.plot(new_vertices[:,0],new_vertices[:,1],new_vertices[:,2],color = fc,alpha = alpha,linewidth = 1)
                        continue

                    vertices_add = np.copy(vertices) 
                    '''works for boundries of 3d polytopes, has to use original vertices'''
                    while len(vertices_add.tolist()) < 4:
                        '''vertices number is < 4, convex hull algorithm can not work for 3d'''
                        vertices_add = np.vstack((vertices_add,vertices_add[0]))
                    hull_3d = True
                    print(names)

                    try:
                        '''if bold boundaries of 3d polytopes, hull can not be QJ option''' 
                        hull = ConvexHull(vertices_add)
                    except:
                        '''always, 2d polygons (coplanar) can not be calculated in 3d.'''
                        hull_3d = False
#                         all_not_3d_names.append(names)
                        hull = ConvexHull(vertices_add,qhull_options = "QJ")
                    print(names, hull_3d)

                    simplices = hull.simplices 
                    org_triangles = [vertices_add[s] for s in simplices]
                    if fcs:
                        fc = fcs[0]
                    if names == sorted(["Mn^{2+}", "gamma-MnOOH"]):
                        fc = "green"
                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = alpha, facecolor=fc,edgecolor = edc,zorder=100)
                    ax.add_collection3d(pc)
                    if not label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if bold_boundary:

                        # if hull_3d:
                        #     lines1 = get_edges_of_polytope_in3D(org_triangles, hull)
                        # else:
                        lines1 = get_edges_of_plane_in3D(new_vertices)
                        # if len(lines1) < 3:
                        #     lines1 = get_edges_of_plane_in3D(new_vertices)
                        # print(lines1)

                        for li in lines1:

                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 1,linewidth=2)
#                 print(all_not_3d_names)

#                     fn=fn+1
#                     print(names)
#                     fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
#                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase2/" +"2d_"+str(fn) )
            if vis_phase3:
                all_1d_names = [['alpha-Mn2O3', 'alpha-MnO2', 'gamma-MnOOH'], ['Mn3O4', 'alpha-Mn2O3', 'gamma-MnOOH'], ['alpha-MnO2', 'beta-MnO2', 'gamma-MnOOH'], ['Mn^{2+}', 'alpha-MnO2', 'gamma-MnOOH'], ['Mn^{2+}', 'beta-MnO2', 'gamma-MnOOH']]
                for names, sharedvers in phaseCoex["phase3"]:
#                     if names in all_1d_names:
#                         continue
                    sharedvers = np.array(sharedvers)
                    print(names)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])
                    vertices = np.hstack([vertices,sharedvers[:,[indexes[2]]]])
                    fc = np.average([colordict[name] for name in names],axis=0)
                    
                    new_vertices = []
                    for i in vertices.tolist():
                        '''around data can not be used for finding boundaries of 3d Polytopes,
                        we need original vertices to plot boundaries of 3d polytopes;
                        in the meanwhile, non-around data not work for 2d boundaries of polygons'''
                        i=np.around(i, decimals=7).tolist()
                        if i not in new_vertices:
                            new_vertices.append(i)
                    vertices = np.array(new_vertices)
                    # print(len(vertices))
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if len(vertices) <= 2: 
                        '''projection in this space is a line'''
#                         all_1d_names.append(names)
                        ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],color = fc,alpha = 1,linewidth = 2)
#                         fn=fn+1
#                         print(names)
#                         fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
#                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase3/" +"label_1d_"+str(fn) )
                        continue
                    if bold_boundary:
                        edges = get_edges_of_plane_in3D(vertices)
                        for li in edges:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha =1,linewidth=3)

                    if len(vertices)<4:
                        '''if only 3 points(triangle), add another vertex point 
                        to fullfill the requirement of convex hull'''
                        vertices = np.vstack((vertices,vertices[0]))
                    hull = ConvexHull(vertices, qhull_options = "QJ")
                
                    simplices = hull.simplices 
                    if fcs:
                        fc = fcs[1]
                    org_triangles = [vertices[s].tolist() for s in hull.simplices]
                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = 0.2, facecolor=fc,edgecolor = edc,zorder=10)
                    ax.add_collection3d(pc)
#                 print(all_1d_names)
#                     fn=fn+1
#                     print(names)
#                     fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
#                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase3/" +"2d_"+str(fn) )

            if vis_phase4:
                for names, sharedvers in phaseCoex["phase4"]:

                    sharedvers = np.array(sharedvers)
                    print(names)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])
                    vertices = np.hstack([vertices,sharedvers[:,[indexes[2]]]])
                    print(len(vertices))
                    print(vertices)
                    fc = np.average([colordict[name] for name in names],axis=0)
                    new_vertices = []
                    for i in vertices.tolist():
                        '''around data can not be used for finding boundaries of 3d Polytopes,
                        we need original vertices to plot boundaries of 3d polytopes;
                        in the meanwhile, non-around data not work for 2d boundaries of polygons'''
                        i=np.around(i, decimals=7).tolist()
                        if i not in new_vertices:
                            new_vertices.append(i)
                    vertices = np.array(new_vertices)
                    print(len(vertices))
                    print(vertices)
                    if fcs:
                        fc = fcs[2]
                    if len(vertices) == 2: 
                        '''projection in this space is a line'''
                        # print()
                        ax.plot(vertices[:,0],vertices[:,1],vertices[:,2], color = fc,alpha = 1,linewidth = 3)
                    elif len(vertices) == 1:
                        ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2], color = fc,alpha = 1)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
#                     fn=fn+1
#                     print(names)
#                     fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
#                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase4/" +"label_1d_"+str(fn) )
            if vis_phase5:
                for names, sharedvers in phaseCoex["phase5"]:
                    sharedvers = np.array(sharedvers)
                    print(names)
                    fc = np.average([colordict[name] for name in names],axis=0)
                    vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])
                    vertices = np.hstack([vertices,sharedvers[:,[indexes[2]]]])
                    # print(len(vertices))
                    # print(vertices)
                    if fcs:
                        fc = fcs[3]
                    ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2], s = 150,color = fc)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
    #                     fn=fn+1
    #                     print(names)
    #                     fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
    #                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase5/" +"0d_"+str(fn) )
            if vis_axis[-1] == "1/R":
                ax.invert_zaxis()
                # plt.show()
                # gaga
            # fig.savefig("C:/Users/jdche/eclipse-workspace/research" + \
            #                     "/myResearch/A220210_pourbaix_surface_energy/phEKslice/" +str(fn) )

        # ax.set_axis_off()
        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # plt.show() 
        return ax
#         plt.show() 
    def get_phase_coexistence_project_2d(
            self,ax,org_limits, yz, xz, xy,label_domains=True,
            bold_boundary=True,
            edc = None, alpha = 0.3, 
            vis_axis=["1/R","PH","E"],
            vis_phase1 = False,
            vis_phase2 = False,
            vis_phase3 = False,
            vis_phase4 = False,
            vis_phase5 = False,
            fcs = None):
        stable_domain_vertices_constraints = self._cp._stable_domain_vertices1
        phaseCoex = self._cp.get_phase_coexistence_region_w_aqueous_constraint()
        
        phas2 = ['delta-K0.21MnO1.87', 'gamma-MnOOH']
        phas3 = ['delta-K0.21MnO1.87', 'gamma-MnOOH','Mn3O4']
        phas4 = ['alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87', 'gamma-MnOOH','Mn3O4']
        phas5 = ['Mn3O4', 'Mn^{2+}', 'alpha-K0.11MnO1.94', 'delta-K0.21MnO1.87', 'gamma-MnOOH']
#         phas3 = ['alpha-K0.11MnO1.94', 'alpha-Mn2O3','alpha-MnO2']
        phas = [sorted(phas2),sorted(phas3),sorted(phas4),sorted(phas5)]
             
        for phass,nnn in zip(phas,list(phaseCoex.keys())):
            for names,ver in phaseCoex[nnn]:

                if phass == names:
                    print(names)
                    print(ver)
                    phaseCoex[nnn] = [[names,ver]]
                    break
                
        eremain = self.elementList.copy()
        for e in self.fixed:
            eremain.remove(e[0])
        eremain.remove("O")
        eremain +=["E","1/R"]
        indexes = []
        for i in vis_axis:
            if i=="PH":
                i="H"
            indexes.append(eremain.index(i))
        print(eremain)
        print(indexes)
        
        # assign colors
#         if not fcs:
        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c

        
        text_font = {'fontname':'Arial', 'size':'8', 'weight':'normal'}


        if vis_phase2:
            for index in range(len(indexes)):
                for names, sharedvers in phaseCoex["phase2"]:
                    sharedvers = np.array(sharedvers)
                    first = sharedvers[:,[indexes[0]]]
                    second = sharedvers[:,[indexes[1]]]
                    third = sharedvers[:,[indexes[2]]]
                    if index == 0:
                        first = list(first)
                        for iii in range(len(first)):
                            first[iii] = [org_limits[0][0]-yz]
                        first = np.array(first)
                    elif index == 1:
                        second = list(second)
                        for iii in range(len(second)):
                            second[iii] = [org_limits[1][0]-xz]
                        second = np.array(second)        
                    elif index == 2:
                        third = list(third)
                        for iii in range(len(third)):
                            third[iii] = [org_limits[2][0]-xy]
                        third = np.array(third)        
                                
                    vertices = np.hstack([first, second])
                    vertices = np.hstack([vertices,third])
                    fc = np.average([colordict[name] for name in names],axis=0)
                    print()
                    print(names)
                    new_vertices = []
                    for i in vertices.tolist():
                        '''around data can not be used for finding boundaries of 3d Polytopes,
                        we need original vertices to plot boundaries of 3d polytopes;
                        in the meanwhile, non-around data not work for 2d boundaries of polygons'''
                        i=np.around(i, decimals=10).tolist()
                        if i not in new_vertices:
                            new_vertices.append(i)
                    new_vertices = np.array(new_vertices)
                    # print("new_vertices",new_vertices)
                    if len(new_vertices) <= 2: 
                        '''projection in this space is a line'''
                        ax.plot(new_vertices[:,0],new_vertices[:,1],new_vertices[:,2],color = fc,alpha = alpha,linewidth = 1)
                        continue
                    
                    vertices_add = np.copy(vertices) 
                    '''works for boundries of 3d polytopes, has to use original vertices'''
                    while len(vertices_add) < 4:
                        '''vertices number is < 4, convex hull algorithm can not work for 3d'''
                        vertices_add = np.vstack((vertices_add,vertices_add[0]))
                    hull_3d = True
                    
                    try:
                        '''if bold boundaries of 3d polytopes, hull can not be QJ option''' 
                        hull = ConvexHull(vertices_add)
                    except:
                        '''always, 2d polygons (coplanar) can not be calculated in 3d.'''
                        hull_3d = False
    #                         all_not_3d_names.append(names)
                        hull = ConvexHull(vertices_add,qhull_options = "QJ")
                    simplices = hull.simplices 
                    org_triangles = [vertices_add[s] for s in simplices]
                    if fcs:
                        fc = fcs[0]
                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = alpha, facecolor=fc,edgecolor = edc,zorder=100)
                    ax.add_collection3d(pc)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if bold_boundary:
                        if hull_3d:
                            lines1 = get_edges_of_polytope_in3D(org_triangles, hull)
                        else:
                            lines1 = get_edges_of_plane_in3D(new_vertices)
                        # if len(lines1) < 3:
                        #     lines1 = get_edges_of_plane_in3D(new_vertices)
                        for li in lines1:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.6,linewidth=1)
    #                 print(all_not_3d_names)
                    
    #                     fn=fn+1
    #                     print(names)
    #                     fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
    #                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase2/" +"2d_"+str(fn) )
        if vis_phase3:
            
            for index in range(len(indexes)):
                for names, sharedvers in phaseCoex["phase3"]:
                    sharedvers = np.array(sharedvers)
                    first = sharedvers[:,[indexes[0]]]
                    second = sharedvers[:,[indexes[1]]]
                    third = sharedvers[:,[indexes[2]]]
                    if index == 0:
                        first = list(first)
                        for iii in range(len(first)):
                            first[iii] = [org_limits[0][0]-yz]
                        first = np.array(first)
                    elif index == 1:
                        second = list(second)
                        for iii in range(len(second)):
                            second[iii] = [org_limits[1][0]-xz]
                        second = np.array(second)        
                    elif index == 2:
                        third = list(third)
                        for iii in range(len(third)):
                            third[iii] = [org_limits[2][0]-xy]
                        third = np.array(third)        
                                
                    vertices = np.hstack([first, second])
                    vertices = np.hstack([vertices,third])
                    fc = np.average([colordict[name] for name in names],axis=0)
                    
                    new_vertices = []
                    for i in vertices.tolist():
                        '''around data can not be used for finding boundaries of 3d Polytopes,
                        we need original vertices to plot boundaries of 3d polytopes;
                        in the meanwhile, non-around data not work for 2d boundaries of polygons'''
                        i=np.around(i, decimals=7).tolist()
                        if i not in new_vertices:
                            new_vertices.append(i)
                    vertices = np.array(new_vertices)
                    # print(len(vertices))
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 40)
                    if len(vertices) <= 2: 
                        '''projection in this space is a line'''
    #                         all_1d_names.append(names)
                        ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],color = fc,alpha = 1,linewidth = 2)
    #                         fn=fn+1
    #                         print(names)
    #                         fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
    #                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase3/" +"label_1d_"+str(fn) )
                        continue
                    if bold_boundary:
                        edges = get_edges_of_plane_in3D(vertices)
                        for li in edges:
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.5,linewidth=1.5)
    
                    if len(vertices)<4:
                        '''if only 3 points(triangle), add another vertex point 
                        to fullfill the requirement of convex hull'''
                        vertices = np.vstack((vertices,vertices[0]))
                    hull = ConvexHull(vertices, qhull_options = "QJ")
                
                    simplices = hull.simplices 
                    if fcs:
                        fc = fcs[1]
                    org_triangles = [vertices[s].tolist() for s in hull.simplices]
                    pc = a3.art3d.Poly3DCollection(org_triangles, \
                         alpha = 0.13, facecolor=fc,edgecolor = edc,zorder=0)
                    ax.add_collection3d(pc)
#                 print(all_1d_names)
#                     fn=fn+1
#                     print(names)
#                     fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
#                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase3/" +"2d_"+str(fn) )

        if vis_phase4:
            for index in range(len(indexes)):
                for names, sharedvers in phaseCoex["phase4"]:
                    sharedvers = np.array(sharedvers)
                    first = sharedvers[:,[indexes[0]]]
                    second = sharedvers[:,[indexes[1]]]
                    third = sharedvers[:,[indexes[2]]]
                    if index == 0:
                        first = list(first)
                        for iii in range(len(first)):
                            first[iii] = [org_limits[0][0]-yz]
                        first = np.array(first)
                    elif index == 1:
                        second = list(second)
                        for iii in range(len(second)):
                            second[iii] = [org_limits[1][0]-xz]
                        second = np.array(second)        
                    elif index == 2:
                        third = list(third)
                        for iii in range(len(third)):
                            third[iii] = [org_limits[2][0]-xy]
                        third = np.array(third)        
                                
                    vertices = np.hstack([first, second])
                    vertices = np.hstack([vertices,third])

                    fc = np.average([colordict[name] for name in names],axis=0)
                    new_vertices = []
                    for i in vertices.tolist():
                        '''around data can not be used for finding boundaries of 3d Polytopes,
                        we need original vertices to plot boundaries of 3d polytopes;
                        in the meanwhile, non-around data not work for 2d boundaries of polygons'''
                        i=np.around(i, decimals=7).tolist()
                        if i not in new_vertices:
                            new_vertices.append(i)
                    vertices = np.array(new_vertices)
                    print(len(vertices))
                    print(vertices)
                    if fcs:
                        fc = fcs[2]
                    if len(vertices) == 2: 
                        '''projection in this space is a line'''
                        # print()
                        ax.plot(vertices[:,0],vertices[:,1],vertices[:,2], color = fc,alpha = 1,linewidth = 3)
                    elif len(vertices) == 1:
                        ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2], color = fc,alpha = 1)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 200)
#                     fn=fn+1
#                     print(names)
#                     fig.savefig("C:/Users/jiadongc/eclipse-workspace/research" + \
#                                 "/myResearch/A220210_pourbaix_surface_energy/figs_phase4/" +"label_1d_"+str(fn) )
        if vis_phase5:
            for index in range(len(indexes)):
                for names, sharedvers in phaseCoex["phase5"]:
                    sharedvers = np.array(sharedvers)
                    first = sharedvers[:,[indexes[0]]]
                    second = sharedvers[:,[indexes[1]]]
                    third = sharedvers[:,[indexes[2]]]
                    if index == 0:
                        first = list(first)
                        for iii in range(len(first)):
                            first[iii] = [org_limits[0][0]-yz]
                        first = np.array(first)
                    elif index == 1:
                        second = list(second)
                        for iii in range(len(second)):
                            second[iii] = [org_limits[1][0]-xz]
                        second = np.array(second)        
                    elif index == 2:
                        third = list(third)
                        for iii in range(len(third)):
                            third[iii] = [org_limits[2][0]-xy]
                        third = np.array(third)        
                                
                    vertices = np.hstack([first, second])
                    vertices = np.hstack([vertices,third])
                # print(len(vertices))
                # print(vertices)
                    if fcs:
                        fc = fcs[3]
                    ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2], s = 150,color = fc)
                    if label_domains:
                        center = np.average(vertices, axis=0)
                        ax.text(center[0],center[1],center[2],"-".join(names),c=fc,ha='center',
                                va='center',**text_font,zorder = 300)
                    
    def get_pourbaix_metal_ion_concen_diagram(self,limits = None,label_domains=True,
                                              show_phi=False,
                                              vis_axis=['PH',"E"],
                                              edc = None, alpha = 0.4,
                                              bold_boundary = False):
#         if self.fixed != None:

        stable_domain_constraints, stable_domain_vertices_constraints=\
            self._cp._stable_domains1, self._cp._stable_domain_vertices1
        
        C=len(self.elementList)
        if "H" in self.elementList:
            hindex = self.elementList.index("H")
        fix1 = [i for i in self.fixed if i[0] in self.elementList]
        fixed2 = [i for i in self.fixed if i[0] not in self.elementList]
        fixVar = {ee:val for ee,val in fixed2}
        
        fixed = fix1
        F = len(fixed)
        fixIndex = [self.elementList.index(fixed[i][0]) for i in range(F)]
        hindex_o = hindex
        for i in fixIndex:

            if i < hindex_o:
                hindex -= 1
        if self.elementList.index("O") < self.elementList.index("H"):
            hindex -= 1
        if limits is None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
            if self.surf:
                limits+=[[0,1]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != "O":
                    newlimits += [limits[i]]
            if len(fixed2) == 0:
                newlimits += limits[C:]
            else:
                if len(self._cp.extra_comp_axis) == 0:
                    if "E" not in fixVar:
                        newlimits += [limits[-2]]
                    if "1/R" not in fixVar:
                        newlimits += [limits[-1]]
                else:
                    if "E" not in fixVar:
                        newlimits += [limits[-1 - len(self._cp.extra_comp_axis)]]
                    for i, els in enumerate(self._cp.extra_comp_axis):
                        if "log"+els not in fixVar:
                            newlimits += [limits[-(len(self._cp.extra_comp_axis)-i)]]                        
        
            limits = newlimits
        print(limits)

        colordict = {}
        jet= plt.get_cmap('turbo') #turbo
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
#             c = np.array([min(ii*1.5, 1) for ii in c.tolist()])
            colordict[e.name] = c

        label_font = {'fontname':'Arial', 'size':'15', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}


        eremain = self.elementList.copy()

        for e in fix1:
            eremain.remove(e[0])
        eremain.remove("O")
#         print(eremain)
        if "E" not in fixVar:
            eremain += ["E"]
        # if "1/R" not in fixVar:
        #     eremain += ["1/R"]
        for el in self._cp.extra_comp_axis:
            if "log"+el not in fixVar:
                eremain += ["log"+el]
        
        indexes = []
        for i in vis_axis:
            if i=="PH":
                i="H"
            indexes.append(eremain.index(i))

        if len(vis_axis) == 2:

            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()

            ax.set_xlim(limits[indexes[0]])

            ax.set_ylim(limits[indexes[1]])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                 grid_color='k', grid_alpha=0.5)
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                else:
                    ax.set_ylim([0,14])
            if vis_axis[0] not in ELEMENTS:
                plt.xlabel(vis_axis[0],**label_font)
            else:
                plt.xlabel("mu" + vis_axis[0],**label_font)
            if vis_axis[1] not in ELEMENTS:
                plt.ylabel(vis_axis[1],**label_font)
            else:
                plt.ylabel("mu" + vis_axis[1],**label_font)
            # plt.rcParams['text.usetex'] = True
            phase1 = ["Ca5(ReO6)2", "Re2O7"]
            for entry, sharedvers in stable_domain_vertices_constraints.items():
                # if entry.name in phase1:
                #     continue

                sharedvers = np.array(sharedvers)
                vertices = np.hstack([sharedvers[:,[indexes[0]]],sharedvers[:,[indexes[1]]]])

                fc = colordict[entry.name]

                new_vertices = np.array(vertices)

                hull = ConvexHull(new_vertices)
                simplices = hull.simplices 
                edges = [new_vertices[s] for s in simplices]
                
                plt.fill(new_vertices[hull.vertices][:,0], new_vertices[hull.vertices][:,1], 
                         alpha = 0.3,facecolor = fc)
                if label_domains:
                    center = np.average(vertices, axis=0)

                    ax.text(center[0],center[1], entry.symbol,c="k",ha='center',
                            va='center',**text_font,zorder = 40)

                if bold_boundary:
                    for li in edges:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],color='k',alpha = 0.6,linewidth=1)
            plt.show()
        if len(vis_axis) == 3:
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=166
            ax.elev=38
            ax.set_xlim(limits[indexes[0]])
            ax.set_ylim(limits[indexes[1]])
            ax.set_zlim(limits[indexes[2]])
            
            if "PH" in vis_axis:
                if vis_axis.index("PH") == 0:
                    ax.set_xlim([0,14])
                elif vis_axis.index("PH") == 1:
                    ax.set_ylim([0,14])
                else:
                    ax.set_zlim([0,14])
            if vis_axis[0] not in ELEMENTS:
                plt.xlabel(vis_axis[0],**label_font)
            else:
                plt.xlabel("mu" + vis_axis[0],**label_font)
            if vis_axis[1] not in ELEMENTS:
                plt.ylabel(vis_axis[1],**label_font)
            else:
                plt.ylabel("mu" + vis_axis[1],**label_font)
            if vis_axis[2] not in ELEMENTS:
                ax.set_zlabel(vis_axis[2],**label_font)
            else:
                ax.set_zlabel("mu" + vis_axis[2],**label_font)
            phases1 = ["CaReNO2","LaTaN2O"]
            for entry in stable_domain_vertices_constraints:
                print(entry.name)
                # if entry.name not in phases1:
                #     continue

                org_vers = stable_domain_vertices_constraints[entry]

                vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])
                vertices = np.hstack([vertices,org_vers[:,[indexes[2]]]])

                try:
                    hull = ConvexHull(vertices)
                except:
                    hull = ConvexHull(vertices,qhull_options = "QJ")
                simplices = hull.simplices 
                org_triangles = [vertices[s] for s in simplices]

                pc = a3.art3d.Poly3DCollection(org_triangles, \
                     alpha = 0.2, facecolor=colordict[entry.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],entry.symbol,c="k",ha='center',
                            va='center',**text_font,zorder = 40)

                if bold_boundary:
                    lines1 = get_edges_of_polytope_in3D(org_triangles, hull)
                
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.6,linewidth=0.5)
            plt.show()

    def get_pourbaix_diagram_diff_T(self,limits = None,label_domains=True,show_phi=False,
                             edc = None, alpha = 0.4, ammonia = None,
                             munh3s = None, muGa3s = None):
#         if self.fixed != None:

        stable_domain_constraints, stable_domain_vertices_constraints=\
            self._cp.get_generalized_chem_pot_domains_w_aqueous_constraints(
                    self._cp._processed_entries, self._cp.elementList, 
                    fixed=self.fixed, mutoPH=self._cp.mutoPH, 
                    limits=limits,
                    show_phi=show_phi,
                    extra_comp_axis = None)
        if ammonia:
            erase_el = "N"
        else:
            erase_el = "O"
        ecopy = self.elementList.copy()
        for e in self.fixed:
            ecopy.remove(e[0])
        ecopy.remove(erase_el)
        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]

        if limits == None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != erase_el:
                    newlimits += [limits[i]]
            newlimits += [limits[-1]]
            limits = newlimits


        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c
            # print(e.name, c)
        
        colordict = {'Mn': [1.0, 0.0, 0.16, 1.0], 'Mn^{2+}': [1.0, 0.2829888712241654, 0.0, 1.0], 'Mn^{3+}': [1.0, 0.7281399046104929, 0.0, 1.0], 'MnO4^{1-}': [0.8055113937466878, 1.0, 0.0, 1.0], 'MnO4^{2-}': [0.8024296675191819, 0.0, 1.0, 1.0], 'Mn(OH)3^{1-}': [1.0, 0.0, 0.75, 1.0], 'Mn(OH)2': [0.0, 1.0, 0.5481762597512125, 1.0], 'Mn3O4': [0.0, 1.0, 0.9909340080118073, 1.0], 'gamma-MnOOH': [0.0, 0.5615942028985503, 1.0, 1.0], 'Mn2O3': [0.0, 0.09271099744245515, 1.0, 1.0], 'beta-MnO2': [0.35485933503836337, 0.0, 1.0, 1.0]}
        colordict = {'GaN': [1.0, 0.0, 0.16, 1.0], 'Ga': [0.8055113937466878, 1.0, 0.0, 1.0], 'Ga^{3+}': [0.0, 1.0, 0.5481762597512125, 1.0], 'MnO4^{2-}': [0.0, 0.09271099744245515, 1.0, 1.0], 'Mn(OH)3^{1-}': [1.0, 0.0, 0.75, 1.0]}
        label_font = {'fontname':'Arial', 'size':'15', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(ecopy) == 3-2 and show_phi == False:
            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.set_xlim(limits[0])
            if self._cp.mutoPH ==True:
                ax.set_xlim([0,14])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])
            nn = 0
            for munh3, muGa3 in zip(munh3s, muGa3s):
                nn += 1
                if nn % 2 == 0:
                    style = "dashed"
                else:
                    style = "solid"

                cpentries = self._cp._processed_entries
                cpentries.append(GeneralizedEntry("Ga^{3+}", muGa3 * 0.01036, self._cp.elementList))
                gpd = GeneralizedDiagram(cpentries,
                                         self._cp.elementList,fixed=self.fixed, 
                                         limits=self._cp.limits,
                                         normalizedfactor="Nm",mutoPH=True,
                                         specific_muh2o=munh3,
                                         ammonia = ammonia)
                stable_domain_constraints, stable_domain_vertices_constraints=\
                gpd.get_generalized_chem_pot_domains_w_aqueous_constraints(
                        gpd._processed_entries, gpd.elementList, 
                        fixed=self.fixed, mutoPH=gpd.mutoPH, 
                        limits=gpd.limits,
                        show_phi=show_phi,
                        extra_comp_axis = None)
                for entry in stable_domain_vertices_constraints:
                    print(entry.name)
    
                    vertices = stable_domain_vertices_constraints[entry][:,:]
                    center = np.average(vertices, axis=0)
                    x, y = np.transpose(np.vstack([vertices, vertices[0]]))
                    plt.fill(x, y, alpha = alpha,facecolor = colordict[entry.name])


                vertices = {entry: [] for entry in stable_domain_constraints}
                '''print edge'''
                linesDict = {entry: [] for entry in stable_domain_constraints}
                for entry in stable_domain_constraints:
    
                    lines = []
                    for tri in stable_domain_constraints[entry]:
                        lines += list(combinations(tri.tolist(),2))
                    lines1 = []
                    for i in lines:
                        if i not in lines1:
                            lines1.append(i)
                        else:
                            lines1.remove(i)
                    linesDict[entry] = lines1
                    
                    for li in lines1:
                        for ver in li:
                            if ver not in vertices[entry]:
                                vertices[entry].append(ver)
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],color="k",linestyle = style,
                                alpha = 0.8,linewidth=1.5)

                    
            if label_domains:
                plt.annotate(entry.symbol, center, ha='center',
                             va='center', **text_font,color = "k",zorder=40)
            if self._cp.mutoPH == True:
                plt.xlabel("PH",**label_font)
            else:
                plt.xlabel('Chem Pot H'+self.elementList[0],**label_font)
            plt.ylabel('E (V)',**label_font)
        plt.show()
    def get_pourbaix_diagram(self,limits = None,label_domains=True,show_phi=False,
                             edc = None, alpha = 0.4, ammonia = None):
#         if self.fixed != None:

        stable_domain_constraints, stable_domain_vertices_constraints=\
            self._cp.get_generalized_chem_pot_domains_w_aqueous_constraints(
                    self._cp._processed_entries, self._cp.elementList, 
                    fixed=self.fixed, mutoPH=self._cp.mutoPH, 
                    limits=limits,
                    show_phi=show_phi,
                    extra_comp_axis = None)
        if ammonia:
            erase_el = "N"
        else:
            erase_el = "O"
        ecopy = self.elementList.copy()
        for e in self.fixed:
            ecopy.remove(e[0])
        ecopy.remove(erase_el)
        F = len(self.fixed)
        C = len(self.elementList)
        fixIndex = [self.elementList.index(self.fixed[i][0]) for i in range(F)]

        if limits == None:
            limits = []
            for i in range(C-F-1):
                limits += [[-10,0]]
            limits += [[-5,5]]
        else:
            newlimits = []
            for i in range(C):
                if i not in fixIndex and self.elementList[i] != erase_el:
                    newlimits += [limits[i]]
            newlimits += [limits[-1]]
            limits = newlimits

        print(limits)
        colordict = {}
        jet= plt.get_cmap('gist_rainbow')
        n = len(stable_domain_vertices_constraints.keys())
        colorr=iter(jet(np.linspace(0,1,n)))
        cpentries = list(stable_domain_vertices_constraints.keys())
        for e in cpentries:
            c = next(colorr)
            colordict[e.name] = c
            # print(e.name, c)
        print(colordict)
        
#         colordict = {'Mn': [1.0, 0.0, 0.16, 1.0], 'Mn^{2+}': [1.0, 0.2829888712241654, 0.0, 1.0], 'Mn^{3+}': [1.0, 0.7281399046104929, 0.0, 1.0], 'MnO4^{1-}': [0.8055113937466878, 1.0, 0.0, 1.0], 'MnO4^{2-}': [0.8024296675191819, 0.0, 1.0, 1.0], 'Mn(OH)3^{1-}': [1.0, 0.0, 0.75, 1.0], 'Mn(OH)2': [0.0, 1.0, 0.5481762597512125, 1.0], 'Mn3O4': [0.0, 1.0, 0.9909340080118073, 1.0], 'gamma-MnOOH': [0.0, 0.5615942028985503, 1.0, 1.0], 'Mn2O3': [0.0, 0.09271099744245515, 1.0, 1.0], 'beta-MnO2': [0.35485933503836337, 0.0, 1.0, 1.0]}
#         colordict = {'GaN': [1.0, 0.0, 0.16, 1.0], 'Ga': [0.8055113937466878, 1.0, 0.0, 1.0], 'Ga^{3+}': [0.0, 1.0, 0.5481762597512125, 1.0], 'MnO4^{2-}': [0.0, 0.09271099744245515, 1.0, 1.0], 'Mn(OH)3^{1-}': [1.0, 0.0, 0.75, 1.0]}
        label_font = {'fontname':'Arial', 'size':'15', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(ecopy) == 3-2 and show_phi == False:
            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.set_xlim(limits[0])
            if self._cp.mutoPH ==True:
                ax.set_xlim([0,14])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[1])
            vertices_dict = {}
            for entry in cpentries:
                print(entry.name)
                # print(stable_domain_vertices_constraints[entry])
                vertices = stable_domain_vertices_constraints[entry][:,:]
                vertices_dict[entry.name] = vertices.tolist()
#                 if entry.name == 'BaN2':
                center = np.average(vertices, axis=0)
#                     print(vertices)
#                     print(entry.name,center)
                x, y = np.transpose(np.vstack([vertices, vertices[0]]))

                # print(colordict[entry.name])
                plt.fill(x, y, alpha = alpha,facecolor = colordict[entry.name])
                
                if label_domains:
                    plt.annotate(entry.symbol, center, ha='center',
                                 va='center', **text_font,color = "k",zorder=40)

            print(vertices_dict)

            if self._cp.mutoPH == True:
                plt.xlabel("PH",**label_font)
            else:
                plt.xlabel('Chem Pot H'+self.elementList[0],**label_font)
            plt.ylabel('E (V)',**label_font)
        elif len(ecopy) == 3-2 and show_phi ==True:
            limits += [[-30,0]]
            

            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            if self._cp.mutoPH == True:
                plt.xlabel("PH",**label_font)
                ax.set_xlim([0,14])
            else:
                
                plt.xlabel('Chem Pot H'+self.elementList[0],**label_font)
            plt.ylabel('E (V)',**label_font)
            ax.set_zlabel('Grand Pot Φ',**label_font)
            
            for entry in stable_domain_constraints:
            
                pc = a3.art3d.Poly3DCollection(stable_domain_constraints[entry], alpha = alpha,
                               facecolor = colordict[entry.name])#,edgecolor = edc)
                ax.add_collection3d(pc)

#                 print(np.array(stable_domain_constraints[entry]))
                projections = []
                for i in stable_domain_constraints[entry]:
                    i[:,-1]=-30
                    projections.append(i)
                pc = a3.art3d.Poly3DCollection(projections, alpha = alpha,
                               facecolor = colordict[entry.name])#,edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(stable_domain_vertices_constraints[entry], axis=0)
                    ax.text(center[0],center[1],center[2],entry.symbol,c='k',ha='center',
                            va='center',**text_font,zorder = 40)
                    center[-1]=-30
                    ax.text(center[0],center[1],center[2],entry.symbol,c='k',ha='center',
                            va='center',**text_font,zorder = 40)
          
        plt.show()


            
        
    def get_generalized_diagram(self,vis_axis=None,limits = None,bold_boundary=False,label_domains=True,edc = None, alpha = 0.5):
#         if self.fixed != None:
        ecopy = self.elementList.copy()
        if limits == None:
            limits = []
            for i in range(len(ecopy)):
                limits += [[-10,0]]
            limits += [[0,5]]
        eremain_index = []
        for el in vis_axis:
            if el in ecopy:
                eremain_index.append(ecopy.index(el))
            else:
                eremain_index.append(-1)
        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(ecopy) == 3-1:
            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            CP_domain_vertices = self._cp._stable_domain_vertices
            colordict = {}
            for e in CP_domain_vertices:
                c = next(color)
                colordict[e.name] = c
#             if show_polytope:
            for e in CP_domain_vertices:

                # if e.name != 'Fe':
                #     continue
                org_vers = CP_domain_vertices[e]
                vertices = np.hstack([org_vers[:,[eremain_index[0]]],org_vers[:,[eremain_index[1]]]])
                vertices = np.hstack([vertices,org_vers[:,[eremain_index[2]]]])
                hull = ConvexHull(vertices)
                simplices = hull.simplices
                org_triangles = [CP_domain_vertices[e][s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, alpha = 0.15, 
                                               facecolor=colordict[e.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    if e.name !="Fe8N":
                        center = np.average(CP_domain_vertices[e], axis=0)
                        ax.text(center[0],center[1],center[2],e.name,ha='center',
                                va='center',**text_font,color = colordict[e.name],zorder=40)
                if bold_boundary:
                    lines = []
                    for tri in org_triangles:
                        lines += list(combinations(tri.tolist(),2))
                    # print(lines[0])
                    lines1 = []
                    for li in lines:
                        v1=li[0]
                        v2=li[1]
                        num_planes_line_shared = 0
                        # print(v1)
                        # print(v2)
                        eqs = hull.equations
                        eqs = np.unique(eqs, axis = 0)
                        for eq in eqs:
                            if abs(np.dot(eq[0:3],v1)+eq[3]) <1e-6 \
                                and abs(np.dot(eq[0:3],v2)+eq[3]) <1e-6:
                                    num_planes_line_shared += 1
                                    # print(abs(np.dot(eq[0:3],v1)+eq[3]),abs(np.dot(eq[0:3],v2)+eq[3]))
                                    # print()
                                    print(eq)
                        # print(num_planes_line_shared)
                        if num_planes_line_shared == 2:
                            lines1.append(li)
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color='k',alpha = 0.5,linewidth=1)
            ax.grid(b=None)
            ax.dist=10
            ax.azim=30
            ax.elev=10
            if vis_axis[0] in ecopy:
                xname='Chem Pot '+vis_axis[0]
            else:
                xname=vis_axis[0]
            if vis_axis[1] in ecopy:
                yname='Chem Pot '+vis_axis[1]
            else:
                yname=vis_axis[1]
            if vis_axis[2] in ecopy:
                zname='Chem Pot '+vis_axis[2]
            else:
                zname=vis_axis[2]
            ax.set_xlabel(xname,fontname='Arial',fontsize = 15)
            ax.set_ylabel(yname,fontname='Arial',fontsize = 15)
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(start, end+0.01,0.5))
            start, end = ax.get_ylim()
            ax.yaxis.set_ticks(np.arange(start, end+0.01,0.5))
            ax.set_zlabel(zname,fontname='Arial',fontsize = 15)
            plt.show()
        return fig,ax
    
    def get_H_xx_diagram(self,limits = None,label_phase2=False,
                         label_phase3=False,label_domains=True,edc = None, alpha = 0.5):
#         if self.fixed != None:
        ecopy = self.elementList.copy()
        if limits == None:
            limits = []
            for i in range(len(ecopy)):
                limits += [[-10,0]]
            limits += [[0,5]]

        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(ecopy) == 2:
            plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.set_xlim([-0.002,1.001])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, 
                           colors='k',grid_color='k', grid_alpha=0.5)
            ax.set_ylim(limits[-1])
#             ax.set_ylim([-12,2])
            jet= plt.get_cmap('gist_rainbow')
            n=len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            colordict = {}
            for e in self._cp._stable_domain_vertices:
                c = next(color)
                colordict[e.name] = c
            if label_domains:
                ax.set_ylabel('H',**label_font)
                ax.set_xlabel("{el1}x{el2}(1-x)".format(el1=self.elementList[0], 
                                                        el2=self.elementList[1]),**label_font)

            stable_domains = self._cp._stable_domains
            stable_domain_vertices = self._cp._stable_domain_vertices
            '''plot a line, single phase'''
            
            for entry, vertices in stable_domain_vertices.items():
                print(entry.name)
                
                mu_interest = vertices[:,-1]
                print([max(mu_interest), min(mu_interest)])
                x1 = Composition(entry.name).get_atomic_fraction(ecopy[0])
                plt.plot([x1,x1],[max(mu_interest), min(mu_interest)],
                         linewidth = 3,color = colordict[entry.name])
                if label_domains:
                    plt.annotate(generate_entry_label(entry), 
                                 [x1,(max(mu_interest)+min(mu_interest))/2], ha='center',
                                 va='center', **text_font,color = colordict[entry.name])
            '''plot 2-phase coexistence'''

            twoPhases = {}
            for entry1, entry2 in list(combinations(stable_domain_vertices.keys(),2)):
                print(entry1.name,entry2.name)
                sharedvers=[]
                for ver in stable_domain_vertices[entry1].tolist():
                    # print(ver)
                    if ver in stable_domain_vertices[entry2].tolist():
                        sharedvers.append(ver)
                if len(sharedvers) != 0:
                    twoPhases["-".join([entry1.name,entry2.name])]=sharedvers
                    x1 = Composition(entry1.name).get_atomic_fraction(ecopy[0])
                    x2 = Composition(entry2.name).get_atomic_fraction(ecopy[0])
                    color=(colordict[entry1.name] + colordict[entry2.name])/2
                    mu_interest = np.array(sharedvers)[:,-1]
                    minn=min(mu_interest)
                    maxx=max(mu_interest)
                    rec=Rectangle((min(x1,x2),minn),abs(x2-x1),maxx-minn,
                                           color=color,alpha=0.3)
                    ax.add_patch(rec)
                    if label_phase2:
                        center = [rec.get_x()+rec.get_width()/2,rec.get_y()+rec.get_height()/2]
                        ax.text(center[0],center[1],"-".join([entry1.name,entry2.name]),c='k',ha='center',
                                va='center',**text_font)

            # for name,
            threePhases = {}
            for phase2name1, phase2name2 in list(combinations(twoPhases,2)):
                sharedvers=[]
                for ver in twoPhases[phase2name1]:
                    if ver in twoPhases[phase2name2]:
                        sharedvers.append(ver)
                if len(sharedvers) !=0:
                    names1 = phase2name1.split("-")
                    names2 = phase2name2.split("-")
                    names=[]
                    for i in names1+names2:
                        if i not in names:
                            names.append(i)
                    names.sort()
                    namesstr = "-".join(names)
                    if namesstr not in threePhases:
                        threePhases[namesstr]=sharedvers
                        x1 = Composition(names[0]).get_atomic_fraction(ecopy[0])
                        x2 = Composition(names[1]).get_atomic_fraction(ecopy[0])
                        x3 = Composition(names[2]).get_atomic_fraction(ecopy[0])
                        mu_interest=np.array(sharedvers)[:,-1]
                        ax.plot([min(x1,x2,x3),max(x1,x2,x3)],[mu_interest[0],mu_interest[1]],
                                c="red",zorder=1,linewidth=1)
                        if label_phase3:
                            center = np.average([[min(x1,x2,x3),max(x1,x2,x3)],[mu_interest[0],mu_interest[1]]], axis=1)
                            print([mu_interest[0],mu_interest[1]],[min(x1,x2,x3),max(x1,x2,x3)])

                            ax.text(center[0],center[1],namesstr,c='r',ha='center',
                                    va='center',**text_font)   
                

        plt.show() 
    
    def get_generalized_diagram_mumur(self,vis_axis = [], limits = None,label_domains=True,
                                      edc = None, alpha = 0.5, title = None):

        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        eremain = self.elementList.copy()
        if self.fixed:
            for e in self.fixed:
                eremain.remove(e[0])
        eremain +=["E","1/R"]

        indexes = []
        for i in vis_axis:
            if i=="PH":
                i="H"
            indexes.append(eremain.index(i))
            
        if len(vis_axis) == 2:
            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = plt.gca()
            ax.set_xlim(limits[1])
            ax.set_ylim(limits[-1])
#             ax.set_xticks([-8,-6,-4,-2,0])
#             ax.set_yticks([-8,-6,-4,-2,0])
            ax.tick_params(direction='out',labelsize= 15, length=2, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
            
            CP_domain_vertices = self._cp._stable_domain_vertices
            colordict = {}
            cpentries = list(CP_domain_vertices.keys())
            cpentries = sorted(cpentries,key = lambda e:Composition(e.pureformula).get_atomic_fraction(self.elementList[0]))
            for e in cpentries:
                c = next(color)
                colordict[e.name] = list(c)
            print(indexes)
            for e in CP_domain_vertices:

                org_vers = CP_domain_vertices[e]

                vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])
                hull = ConvexHull(vertices)

                plt.fill(vertices[hull.vertices][:,0], vertices[hull.vertices][:,1], 
                         alpha = 0.3,facecolor = colordict[e.name])

                if label_domains:
                    center = np.average(vertices, axis=0)
                    plt.annotate(generate_entry_label(e), center, ha='center',
                                 va='center', **text_font,color = colordict[e.name],
                                 zorder=40)

            plt.xlabel(vis_axis[0],**label_font)
            plt.ylabel(vis_axis[1],**label_font)
        
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
            
            if vis_axis[-1] == "1/R":
                ax.invert_yaxis()        
        if len(vis_axis) == 3:
            jet= plt.get_cmap('rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            CP_domain_vertices = self._cp._stable_domain_vertices
            colordict = {}
            cpentries = list(CP_domain_vertices.keys())
            cpentries = sorted(cpentries,key = lambda e:Composition(e.pureformula).get_atomic_fraction(self.elementList[0]))
            for e in cpentries:
                c = next(color)
                colordict[e.name] = list(c)

            for e in CP_domain_vertices:
                print(e.name)
                org_vers = CP_domain_vertices[e]
                vertices = np.hstack([org_vers[:,[indexes[0]]],org_vers[:,[indexes[1]]]])
                vertices = np.hstack([vertices,org_vers[:,[indexes[2]]]])
                hull = ConvexHull(vertices)
                simplices = hull.simplices
                org_triangles = [vertices[s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, alpha = 0.15, 
                                               facecolor=colordict[e.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(vertices, axis=0)
                    ax.text(center[0],center[1],center[2],e.name,ha='center',
                            va='center',**text_font,color = colordict[e.name],zorder=40)
            
            ax.grid(b=None)
            ax.dist = 10
            for az in range(5, 80):
                ax.azim = az
                ax.elev = 18
    
                ax.set_xlabel('Chem Pot '+eremain[indexes[0]],fontname='Arial',fontsize = 15)
                ax.set_ylabel('Chem Pot '+eremain[indexes[1]],fontname='Arial',fontsize = 15)
                ax.set_zlabel(eremain[indexes[2]],fontname='Arial',fontsize = 15)
                
                ax.set_xlim(limits[indexes[0]])
                ax.set_ylim(limits[indexes[1]])
                ax.set_zlim(limits[indexes[2]])
    
                if title is not None:
                    ax.text(2, -2, -3.5, title, fontname='Arial',fontsize = 15)
                if vis_axis[-1] == "1/R":
                    ax.invert_zaxis()
                plt.savefig(str(az) + "haha.png")
        return fig,ax
    
    def get_generalized_diagram_slice(self,limits = None,label_domains=True,
                                      edc = None, alpha = 0.5, title = None):
#         if self.fixed != None:
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
            limits += [[-5,5]]
        else:
            newlimits = []
            for i in range(C+1):
                if i not in fixIndex:
                    newlimits += [limits[i]]
            limits = newlimits
            print(limits)

        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(ecopy) == 3-1:
            jet= plt.get_cmap('gist_rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            CP_domain_vertices = self._cp._stable_domain_vertices
            colordict = {}
            for e in CP_domain_vertices:
                print(e.name)
                c = next(color)
                colordict[e.name] = list(c)
#             print(colordict)
#             colordict = {'La^{3+}': [1.0, 0.0, 0.16, 1.0], 'TaO2^{1+}': [1.0, 0.36777954425013254, 0.0, 1.0], 'LaH3': [1.0, 0.918918918918919, 0.0, 1.0], 'La2O3': [0.5511393746687864, 1.0, 0.0, 1.0], 'LaTa3O9': [0.0, 1.0, 0.0, 1.0], 'LaTaO4': [0.0, 1.0, 0.5481762597512125, 1.0], 'LaTa7O19': [0.0, 0.9239130434782604, 1.0, 1.0], 'La3TaO7': [0.0, 0.36977834612105687, 1.0, 1.0], 'Ta2O5': [0.16304347826086973, 0.0, 1.0, 1.0], 'TaO10': [0.7171781756180736, 0.0, 1.0, 1.0], 'LaTaN2O': [1.0, 0.0, 0.75, 1.0]}


#             if show_polytope:
            
            for e in CP_domain_vertices:

                # if e.name != 'LaTaN2O':
                #     continue
                hull = ConvexHull(CP_domain_vertices[e])
                simplices = hull.simplices
                org_triangles = [CP_domain_vertices[e][s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, alpha = 0.15, 
                                               facecolor=colordict[e.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(CP_domain_vertices[e], axis=0)
                    ax.text(center[0],center[1],center[2],e.symbol,ha='center',
                            va='center',**text_font,color = colordict[e.name],zorder=40)
            
            CPtriangles = self._cp._stable_domains
            CPtriangles_inslice = {entry:[] for entry in self._cp._stable_domains}
            if self._cp.slicePlane:
                for e in CPtriangles:
                    for tri in CPtriangles[e]:
                        if is_tri_in_planes(tri,self._cp.slicelist):
                            CPtriangles_inslice[e].append(tri)

                '''print edge'''
                linesDict = {entry: [] for entry in CPtriangles_inslice}
                for entry in CPtriangles_inslice:
    
                    lines = []
                    for tri in CPtriangles_inslice[entry]:
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
                    for li in lines1:
                        li = np.array(li)
                        ax.plot(li[:,0],li[:,1],li[:,2],color="k",alpha = 0.8,linewidth=2)
                        
            
            
            
            ax.grid(b=None)
            ax.dist = 10
            ax.azim = -117
            ax.elev = 29
#             ax.azim=315
#             ax.elev=-12
            ax.set_xlabel('Chem Pot '+ecopy[0],fontname='Arial',fontsize = 15)
            ax.set_ylabel('Chem Pot '+ecopy[1],fontname='Arial',fontsize = 15)
#             ax.set_title('subplot 2', label_font, loc = "right")
            phwithtitle = 'PH'
#             phwithtitle += "     muN = " + self._cp.fixed[-1][-1]
            if self._cp.mutoPH == True:
                limits[ecopy.index("H")]=[0,14]
                if ecopy.index("H") == 1:
                    ax.set_ylabel(phwithtitle,fontname='Arial',fontsize = 15)
                if ecopy.index("H") == 0:
                    ax.set_xlabel(phwithtitle,fontname='Arial',fontsize = 15)
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            ax.set_zlabel('E (V)',fontname='Arial',fontsize = 15)
            if title is not None:
                ax.text(2, -2, -3.5, title, fontname='Arial',fontsize = 15)
        return fig,ax

    def get_generalized_diagram_slice_diff_t_p(self,muh2os,limits = None,label_domains=True,
                                      edc = None, alpha = 0.5, title = None,label_pourbaix = 0,
                                      ammonia = None):
#         if self.fixed != None:
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
            limits += [[-5,5]]
        else:
            newlimits = []
            for i in range(C+1):
                if i not in fixIndex:
                    newlimits += [limits[i]]
            limits = newlimits
            print(limits)

        label_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal'}
        text_font = {'fontname':'Arial', 'size':'15', 'weight':'normal'}
        if len(ecopy) == 3-1:
            jet= plt.get_cmap('gist_rainbow')
            n = len(self._cp._stable_domain_vertices.keys())
            n += 2
#             n = len(self._cp._processed_entries)
            color=iter(jet(np.linspace(0,1,n)))
            fig = plt.figure(figsize=(9.2, 7))
            ax = a3.Axes3D(fig) 
            CP_domain_vertices = self._cp._stable_domain_vertices
            colordict = {}
            for e in self._cp._stable_domain_vertices:
#                 print(e.name)
                c = next(color)
                colordict[e.name] = list(c)
            c = next(color)
            colordict["MnO4^{2-}"] = list(c)
            c = next(color)
            colordict["Mn(OH)3^{1-}"] = list(c)
            colordict = {'Mn': [1.0, 0.0, 0.16, 1.0], 'Mn^{2+}': [1.0, 0.2829888712241654, 0.0, 1.0], 'Mn^{3+}': [1.0, 0.7281399046104929, 0.0, 1.0], 'MnO4^{1-}': [0.8055113937466878, 1.0, 0.0, 1.0], 'MnO4^{2-}': [0.8024296675191819, 0.0, 1.0, 1.0], 'Mn(OH)3^{1-}': [1.0, 0.0, 0.75, 1.0], 'Mn(OH)2': [0.0, 1.0, 0.5481762597512125, 1.0], 'Mn3O4': [0.0, 1.0, 0.9909340080118073, 1.0], 'gamma-MnOOH': [0.0, 0.5615942028985503, 1.0, 1.0], 'Mn2O3': [0.0, 0.09271099744245515, 1.0, 1.0], 'beta-MnO2': [0.35485933503836337, 0.0, 1.0, 1.0]}
            colordict = {'GaN': [1.0, 0.0, 0.16, 1.0], 'Ga': [0.8055113937466878, 1.0, 0.0, 1.0], 'Ga^{3+}': [0.0, 1.0, 0.5481762597512125, 1.0], 'MnO4^{2-}': [0.0, 0.09271099744245515, 1.0, 1.0], 'Mn(OH)3^{1-}': [1.0, 0.0, 0.75, 1.0]}

            for e in CP_domain_vertices:

                hull = ConvexHull(CP_domain_vertices[e])
                simplices = hull.simplices
                org_triangles = [CP_domain_vertices[e][s] for s in simplices]
                pc = a3.art3d.Poly3DCollection(org_triangles, alpha = 0.1, 
                                               facecolor=colordict[e.name],edgecolor = edc)
                ax.add_collection3d(pc)
                if label_domains:
                    center = np.average(CP_domain_vertices[e], axis=0)
                    ax.text(center[0],center[1],center[2],e.symbol,ha='center',
                            va='center',**text_font,color = colordict[e.name],zorder=40)
            

            for muh2o in muh2os:

                gpd = GeneralizedDiagram(self._cp._processed_entries,
                                        self._cp.elementList,fixed=self.fixed, 
                                        limits=self._cp.limits,
                                        normalizedfactor="Nm",mutoPH=True,
                                        slicePlane=1,sliceLower=False,
                                        specific_muh2o = muh2o,
                                        ammonia = ammonia)

                CPtriangles = gpd._stable_domains
                CPtriangles_inslice = {entry:[] for entry in CPtriangles}
                if gpd.slicePlane:
                    for e in CPtriangles:
                        for tri in CPtriangles[e]:
                            if is_tri_in_planes(tri,gpd.slicelist,muh2o):
                                CPtriangles_inslice[e].append(tri)
                    vertices = {entry: [] for entry in CPtriangles_inslice}
                    '''print edge'''
                    linesDict = {entry: [] for entry in CPtriangles_inslice}
                    for entry in CPtriangles_inslice:
        
                        lines = []
                        for tri in CPtriangles_inslice[entry]:
                            lines += list(combinations(tri.tolist(),2))
                        lines1 = []
                        for i in lines:
                            if i not in lines1:
                                lines1.append(i)
                            else:
                                lines1.remove(i)
                        linesDict[entry] = lines1

                        for li in lines1:
                            for ver in li:
                                if ver not in vertices[entry]:
                                    vertices[entry].append(ver)
                            li = np.array(li)
                            ax.plot(li[:,0],li[:,1],li[:,2],color="k",alpha = 0.8,linewidth=1.5)
                            
                    vertices = {k: v for k, v in vertices.items() if v}


                    for entry in vertices:
                        vertices[entry] = np.array(vertices[entry])
                        vertices_add = np.copy(vertices[entry]) 
                        '''works for boundries of 3d polytopes, has to use original vertices'''
                        while len(vertices_add.tolist()) < 4:
                            '''vertices number is < 4, convex hull algorithm can not work for 3d'''
                            vertices_add = np.vstack((vertices_add,vertices_add[0]))
                        print(entry.name)
                        hull = ConvexHull(vertices_add,qhull_options='QJ')
 
                        simplices = [vertices_add[indices] for indices in hull.simplices]
                         
                        pc = a3.art3d.Poly3DCollection(simplices, alpha = 0.1,
                                       facecolor = colordict[entry.name])#,edgecolor = edc)
                        ax.add_collection3d(pc)
         
 
                        if label_pourbaix:
                            center = np.average(vertices[entry], axis=0)
                            ax.text(center[0],center[1],center[2],entry.symbol,c='k',ha='center',
                                    va='center',**text_font,zorder = 40)
                            center[-1]=-30
                            ax.text(center[0],center[1],center[2],entry.symbol,c='k',ha='center',
                                    va='center',**text_font,zorder = 40)

            ax.grid(b=None)
            ax.dist = 10
            ax.azim = -40
            ax.elev = -10

            ax.set_xlabel('Chem Pot '+ecopy[0],fontname='Arial',fontsize = 15)
            ax.set_ylabel('Chem Pot '+ecopy[1],fontname='Arial',fontsize = 15)

            phwithtitle = 'PH'

            if self._cp.mutoPH == True:
                limits[ecopy.index("H")]=[0,14]
                if ecopy.index("H") == 1:
                    ax.set_ylabel(phwithtitle,fontname='Arial',fontsize = 15)
                if ecopy.index("H") == 0:
                    ax.set_xlabel(phwithtitle,fontname='Arial',fontsize = 15)
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_zlim(limits[2])
            ax.set_zlabel('E (V)',fontname='Arial',fontsize = 15)
            if title is not None:
                ax.text(2, -2, -3.5, title, fontname='Arial',fontsize = 15)
        return fig,ax

    
    def add_aqueous_plane(self,fig,ax):
#         ax = a3.Axes3D(fig)
        PH = np.arange(0,14.01,0.1)
        E = np.arange(-2,2.01,0.1)
        PH,E = np.meshgrid(PH,E)
        muo = MU_H2O+2*self.prefac*PH+2*E
        ax.plot_surface(muo, PH, E, alpha=0.2)
        plt.show()
#         print(PH)
    
    
    def domain_vertices(self, entry):
        """
        Returns the vertices of the Pourbaix domain.

        Args:
            entry: Entry for which domain vertices are desired

        Returns:
            list of vertices
        """
        return self._cp._stable_domain_vertices[entry]

def is_tri_in_planes(tri,sliceplane,muh2o = MU_H2O):
    result = True
    for ver in tri:
        # if abs(abs(np.dot(ver,sliceplane[0:3]))-abs(muh2o)) > 1e-6:
        if abs(abs(np.dot(ver,sliceplane[0:3]))-abs(sliceplane[-1])) > 1e-6:
            result = False
            break 
    return result

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
    # ggaa
    lines1 = []
    for li in lines:
        v1=li[0]
        v2=li[1]
        num_planes_line_shared = 0

        eqs = hull.equations
        eqs = np.unique(eqs, axis = 0)
        for eq in eqs:
            
            if abs(np.dot(eq[0:3],v1)+eq[3]) <1e-3 \
                and abs(np.dot(eq[0:3],v2)+eq[3]) <1e-3:
                    num_planes_line_shared += 1
        # print(v1, v2)
        # print(num_planes_line_shared)
        if num_planes_line_shared == 2:
            '''the edge lines shared by two hyperplanes'''
            # print(abs(np.dot(eq[0:3],v1)+eq[3]),abs(np.dot(eq[0:3],v2)+eq[3]))
            if li not in lines1:
                lines1.append(li)

    return lines1

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
    low_hull = ConvexHull(lowD_vertices)
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

def compare_vertices(v1, v2, n1=0,n2=1):
    '''determine if 2 vertices x,y values are the same, if so, return True
        Args: v1, v2 : list of 3 number
          eg v1 = [1,1,1] '''

    if abs(v1[n1]-v2[n1])>1e-6 or abs(v1[n2]-v2[n2])>1e-6:
        return False # If the content is not equal, return false

    return True  
    
def generate_entry_label(entry):
    return entry.name
