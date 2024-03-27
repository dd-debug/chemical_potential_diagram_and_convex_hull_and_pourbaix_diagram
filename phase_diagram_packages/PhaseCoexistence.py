'''
Created on 2020.09.02

@author: dd
'''
from pymatgen.core.composition import Composition
import numpy as np
from itertools import combinations
class Phase1():
    def __init__(self, eql, org_triangles,projIndex=None, nonProjIndex=None):
        '''tridict has the triangle of each phase to plot in mu-mu-x space
           new_vertices has the vertices of each phase in mu-mu-x space'''
        self.projIndex = projIndex
        self.nonProjIndex = nonProjIndex
        self.elementList = eql.elementList
        triDict = {entry: [] for entry in eql._processed_entries}
        for entry, vertices in eql._stable_domain_vertices.items():
            for tri in org_triangles:
                if all(x in vertices.tolist() for x in tri.tolist()):
                    triDict[entry].append(tri.tolist())
        self.vertices = eql._stable_domain_vertices
        self.triDict = triDict
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
        self.linesDict = linesDict
    def get_comp_after_fix_mu(self, entry):
        if len(entry.composition.elements) == 1 and str(entry.composition.elements[0]) == self.elementList[self.projIndex[0]]:
            return None
        atf = entry.composition.get_atomic_fraction(self.elementList[self.projIndex[0]])
        x1 = entry.composition.get_atomic_fraction(self.elementList[self.nonProjIndex[0]])/(1-atf)
        return x1
        
        
        
        
        
        
        
class Phase23():
    '''class to create a 2/3-phase coexistence region
    Args: triangles to plot
          color
          2 phases name: list of two str'''
    def __init__(self, triangles=None, names = None, color=None):
        self.triangles = triangles
        self.color = color
        self.names = names
        self.vertices=[]
        
        vertices = []
        for tri in triangles:
            for i in tri:
                if i not in vertices:
                    vertices.append(i)
        self.vertices=vertices


def compare_xy_lines(line1, line2,n=2):
    '''determine if 2 lines x,y values are the same, if so, return True
       Args: line1, line2 : list of 2 vertices, each vertices is a list of 3 number
          eg lines1 = [[1,1,1],[2,2,2]] '''
    if len(line1) != len(line2):
        return False # If row numbers are not same, return false

    for row_number in range(len(line1)):
        if line1[row_number][:n] != line2[row_number][:n]:
            return False # If the content is not equal, return false

    return True # All of the content is equal, the return true
def compare_xy_vertices(v1, v2, v3, n=2):
    '''determine if 3 vertices x,y values are the same, if so, return True
       Args: v1, v2,v3 : list of 3 number
          eg v1 = [1,1,1] '''
    if len(v1) != len(v2) or len(v1) != len(v3):
        return False # If row numbers are not same, return false

    if v1[:n] != v2[:n] or v1[:n] != v3[:n]:
        return False # If the content is not equal, return false

    return True # All of the content is equal, the return true

