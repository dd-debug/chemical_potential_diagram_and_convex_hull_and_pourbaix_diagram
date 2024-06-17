# -*- coding: utf-8 -*-
'''
Created on Jun 5, 2021

@author: jiadongc@umich.edu
'''
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter,\
    CompoundPhaseDiagram
import plotly.graph_objects as go
import json
import math
import itertools
import numpy as np
from scipy.interpolate import griddata
from pymatgen.util.string import htmlify
from phase_diagram_packages.EquilibriumLine import EquilLine
from phase_diagram_packages.ChemicalPotentialDiagram import ChemPotDiagram,ChemPotPlotter,trans_PD_to_ChemPot_entries

from pymatgen.core.composition import Composition
from itertools import combinations
import os

from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.ext.matproj import MPRester
# with open("C:/Users/jiadongc/anaconda3/envs/my_pymatgen/lib/site-packages/pymatgen/util/plotly_pd_layouts.json", "r") as f:
#     plotly_layouts = json.load(f)
# with open("C:/Users/jdche/eclipse-workspace/research/myResearch/A240130_mixing_function_of_any_two_ions/plotly_pd_layouts.json", "r") as f:
#     plotly_layouts = json.load(f)
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

with open(current_directory + "/plotly_pd_layouts.json","r") as f:
    plotly_layouts = json.load(f)

def getOrigStableEntriesList(els,filename = None):
    '''save stable entries of els system from materials project in to json files'''
    directory = os.path.join(os.path.dirname(current_directory)) + "/entries_data"
    s_els = list(els).copy()
    s_els.sort()
    if filename == None:
        filename = '-'.join(s_els)
    cache = os.path.join(directory, filename)
    if os.path.exists(cache):
        print('loading from cache.','-'.join(s_els))
        with open(cache, 'r') as f:
            dict_entries = json.load(f)
        list_entries = []
        for e in dict_entries:
            list_entries.append(ComputedEntry.from_dict(e))
        return list_entries
    else:
        print('Reading from database.','-'.join(s_els))
        with MPRester("SZXJWLvi8njBGvA4sT") as MPR:
            entries = MPR.get_entries_in_chemsys(s_els)
        pd = PhaseDiagram(entries)
        newentries=[]
        for e in pd.stable_entries:
            newentries.append(e)
        dict_entries = []
        for e in newentries:
            dict_entries.append(e.as_dict())
        with open(cache,'w') as f:
            json.dump(dict_entries,f)
        return newentries
    
def get_color_tri(p):
    colors = [[0, 0, 1],[1, 0, 0],[0, 1, 0]]
      
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

def triangular_coord(coord):
    """
    Convert a 2D coordinate into a triangle-based coordinate system for a
    prettier phase diagram.

    Args:
        coord: coordinate used in the convex hull computation.

    Returns:
        coordinates in a triangular-based coordinate system.
    """
    unitvec = np.array([[1, 0], [0.5, math.sqrt(3) / 2]])

    result = np.dot(np.array(coord), unitvec)
    return result.transpose()
def tet_coord(coord):
    """
    Convert a 3D coordinate into a tetrahedron based coordinate system for a
    prettier phase diagram.

    Args:
        coord: coordinate used in the convex hull computation.

    Returns:
        coordinates in a tetrahedron-based coordinate system.
    """
    unitvec = np.array(
        [
            [1, 0, 0],
            [0.5, math.sqrt(3) / 2, 0],
            [0.5, 1.0 / 3.0 * math.sqrt(3) / 2, math.sqrt(6) / 3],
        ]
    )
    result = np.dot(np.array(coord), unitvec)
    return result.transpose()

class new_PDPlotter(PDPlotter):
    def get_plot(
        self,
        label_stable=True,
        label_unstable=True,
        ordering=None,
        energy_colormap=None,
        process_attributes=False,
        plt=None,
        add_terminal_plane = False,
        add_tangent_plane = False,
        marksize = 10,
        label_uncertainties=False,
        irpd = None,
        show_label = True,
        scatters = None,
        show_elements = True,
        add_3in4 = None,
        add_triangle_colormap = False
    ):
        """
        :param label_stable: Whether to label stable compounds.
        :param label_unstable: Whether to label unstable compounds.
        :param ordering: Ordering of vertices (matplotlib backend only).
        :param energy_colormap: Colormap for coloring energy (matplotlib backend only).
        :param process_attributes: Whether to process the attributes (matplotlib
            backend only).
        :param plt: Existing plt object if plotting multiple phase diagrams (
            matplotlib backend only).
        :param label_uncertainties: Whether to add error bars to the hull (plotly
            backend only). For binaries, this also shades the hull with the
            uncertainty window.
        :param add_3in4: add a colored ternary compound convex hull in a quanternary convex hull
        :param add_tangent_plane: add a tangent plane of a phase on a convex hull
        :param irpd: add a reaction compound convex hull slice with kinks
        :return: go.Figure (plotly) or matplotlib.pyplot (matplotlib)
        """
        fig = None

        if self.backend == "plotly":
            data = [self._create_plotly_lines()]
            para = 2.5
            if add_triangle_colormap:
                data += self.create_ternary_in_quanternary_hull_triangle_colormap()
            if self._dim == 3:
#                 data.append(self._create_plotly_ternary_support_lines(para = para))
                data += self._create_plotly_ternary_support_lines2(para = para)
                data.append(self._create_plotly_ternary_hull())

            stable_labels_plot = self._create_plotly_stable_labels(label_stable)

            ss = marksize
            stable_marker_plot, unstable_marker_plot = self._create_plotly_markers(
                label_uncertainties, marksize=ss, colorblue = False
            )

            if self._dim == 2 and label_uncertainties:
                data.append(self._create_plotly_uncertainty_shading(stable_marker_plot))
            if show_label:
                data.append(stable_labels_plot)
            data.append(unstable_marker_plot)
#             print(unstable_marker_plot)
            data.append(stable_marker_plot)
            if add_3in4:
                data += self.create_ternary_in_quanternary_hull(terminal_comp = add_3in4)

            '''add triangle and line for llto'''
#             data += self.add_triangle()
#             data += self.add_line_from_target()
            '''add line'''
#             linelist = self.add_line()
#             data += linelist
            '''tangent plane'''
            if add_tangent_plane:
                data += self.create_tangent_plane(add_tangent_plane)
            '''add planes from a terminal comp'''
            if add_terminal_plane:
                data += self.create_reaction_plane2()
            '''add irpd'''
            if irpd:
                for i, ir in enumerate(irpd):
                    comp1 = ir[0]
                    comp2 = ir[1]
                    color = None
                    alpha = 0.3
                    if i == len(irpd) - 1:
                        color = ["#cc33ff","#e699ff"]#["#2eb82e", "#c2f0c2"]#["#ff5c33", "#ffd6cc"]
                        color = ["#2eb82e", "#c2f0c2"]
                        m = 10
                        alpha = 0.5

                    elif i == 1:
                        color = ["#4d94ff","#b3d1ff"]
                        m = 8
                    elif i == 0:
                        color = ["#2eb82e", "#c2f0c2"]
                        color = ["red", "red"]
#                         color = ["#ff5c33", "#ffd6cc"]
                        m = 8
                    elif i == 2:
                        color = ["#ff5c33", "#ffd6cc"]
                        m = 8
                    data += self.create_reaction_plane3([comp1, comp2], color = color,
                                marksize = 6, linewidth = 7, para = 1.5, alpha = alpha)
                    print(comp1, comp2)

                    data += self.get_irpd(comp1, comp2, color = color,
                                marksize = 6, linewidth = 7)
            
            if scatters:
                data += self.add_scatter(scatters[0], scatters[1])

            fig = go.Figure(data=data)
            fig.layout = self._create_plotly_figure_layout(label_stable = show_elements)

        elif self.backend == "matplotlib":
            if self._dim <= 3:
                fig = self._get_2d_plot(
                    label_stable,
                    label_unstable,
                    ordering,
                    energy_colormap,
                    plt=plt,
                    process_attributes=process_attributes,
                )
            elif self._dim == 4:
                fig = self._get_3d_plot(label_stable)
        fig.write_html("test_sample.html")
        return fig
    
    def get_surface_plot(self, label_stable = True, 
                         marksize = 10, show_elements = True,
                         show_label = True,
                         label_uncertainties=False):
        fig = None

        data = []
        data = [self._create_plotly_lines()]
        para = 1.5

        if self._dim == 3:
            data.append(self._create_plotly_ternary_support_lines(para = para))
            data.append(self._create_plotly_ternary_hull())

        stable_labels_plot = self._create_plotly_stable_labels(label_stable)
        stable_marker_plot, unstable_marker_plot = self._create_plotly_markers(
            label_uncertainties, marksize=marksize, colorblue = False
        )

        if show_label:
            data.append(stable_labels_plot)
        data.append(unstable_marker_plot)
        data.append(stable_marker_plot)
        x = unstable_marker_plot["x"] + stable_marker_plot["x"]
        y = unstable_marker_plot["y"] + stable_marker_plot["y"]
        z = unstable_marker_plot["z"] + stable_marker_plot["z"]
        xi = np.linspace(0, 1, 500)
        yi = np.linspace(0, 1, 500)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        
        # 鍒涘缓 Surface 鍥捐〃
        trace = go.Surface(z=zi, x=xi, y=yi, 
                           colorscale='Portland',
                           name='Surface Data',
                           showlegend=True)
        data.append(trace)

        
        fig = go.Figure(data=data)
        fig.layout = self._create_plotly_figure_layout(label_stable = show_elements)

        fig.show()
        return fig  
        
    def add_line_from_target(self, target = "Li0.3La0.57Ti1O3.005", 
                             diff = -0.0966):
        
        stableEntries = self.pd_plot_data[1]
        pd = self._pd
        data = []
        for line in stableEntries:
            entry = stableEntries[line]
            if entry.original_entry.name == target:
                energy = pd.get_form_energy_per_atom(entry)
                compinfo = line
        data.append(go.Scatter3d(
            x=[compinfo[1],compinfo[1]], y=[compinfo[0],compinfo[0]], z=[energy-diff,energy],
                marker=dict(
                    size=0,
                    opacity=0.5,
                    color="#4d94ff",
                    colorscale='Viridis',
                ),
            line=dict(
                color="lightblue",
                width=5
            )))
        return data
    

    
    def add_triangle(self):
        stableEntries = self.pd_plot_data[1]
        pd = self._pd
        data = []
        for line in stableEntries:
#             if stableEntries[line].name in ["NiO2", "CoO","Cr2O3"]:
            if stableEntries[line].original_entry.name in ["TiO2", "La2O3","Li4Ti5O12"]:
#             if stableEntries[line].original_entry.name in ["TiO2", "La2O3","Li2TiO3"]:
                point = []
                point.extend(line)
                point.append(pd.get_form_energy_per_atom(stableEntries[line]))

                data.append(point)
        facets = [0,1,2]
        data1 = []
        data = np.array(data)
        # i, j and k give the vertices of triangles
        # here we represent the 2 triangles of the rectangle
        data1.append(go.Mesh3d(
            x=list(data[:, 1]),
            y=list(data[:, 0]),
            z=list(data[:, 2]),
            i=[facets[1]],
            j=[facets[0]],
            k=[facets[2]],
            opacity=0.8,
            color = "lightblue",
            hoverinfo="none",
            lighting=dict(diffuse=0.0, ambient=1.0),
            name="Convex Hull (shading)",
            flatshading=True,
            showlegend=True,
        ))
        return data1
                

    def add_line(self,marksize=10,linewidth=8):
        els1str = [str(i) for i in self._pd.elements]
        els1 = combinations(els1str,2)
        
        data1 = []
        vertices = []
        for els in els1:
            
            entries = getOrigStableEntriesList(els)
            pd = PhaseDiagram(entries)
            deepest = 0
            for e in entries:
                formE = pd.get_form_energy_per_atom(e)
                if deepest >= formE:
                    deepest = formE
                    deepestentry = e
            
            ver=np.append(triangular_coord([deepestentry.composition.get_atomic_fraction(self._pd.elements[1]),
                              deepestentry.composition.get_atomic_fraction(self._pd.elements[2])]),deepest)

            vertices.append(ver.tolist())
            print(vertices)
            
        vertices = np.array(vertices)
        print(len(vertices))

        for i in range(0,len(vertices)):
            if abs(vertices[i][2] + 0.8471310825000005) < 1e-6:
                c="red"
            else:
                c="blue"
            data1.append(go.Scatter3d(
                x=[vertices[i][1],vertices[i][1]], y=[vertices[i][0],vertices[i][0]], z=[vertices[i][2],0],
#                 marker=dict(
#                     size=0,
#                     opacity=0.5,
#                     color="#4d94ff",
#                     colorscale='Viridis',
#                 ),
                line=dict(
                    color=c,
                    width=linewidth
                )))
        return data1


#     def _create_plotly_element_annotations(self):
#         """
#         Creates terminal element annotations for Plotly phase diagrams.
#   
#         Returns:
#             list of annotation dicts.
#         """
#         annotations_list = []
#         x, y, z = None, None, None
#   
#         for coords, entry in self.pd_plot_data[1].items():
#             if not entry.composition.is_element:
#                 continue
#   
#             x, y = coords[0], coords[1]
#   
#             if self._dim == 3:
#                 z = self._pd.get_form_energy_per_atom(entry)
#             elif self._dim == 4:
#                 z = coords[2]
#   
#             if entry.composition.is_element:
#                 clean_formula = str(entry.composition.elements[0])
#                 if hasattr(entry, "original_entry"):
#                     orig_comp = entry.original_entry.composition
#                     clean_formula = htmlify(orig_comp.reduced_formula)
#   
#                 font_dict = {"color": "#000000", "size": 24.0}
#                 opacity = 1.0
#   
#             annotation = plotly_layouts["default_annotation_layout"].copy()
#             annotation.update(
#                 {
#                     "x": x,
#                     "y": y,
#                     "font": font_dict,
# #                     "text": clean_formula,
#                     "text": "",
#                     "opacity": opacity,
#                 }
#             )
#   
#             if self._dim in (3, 4):
#                 for d in ["xref", "yref"]:
#                     annotation.pop(d)  # Scatter3d cannot contain xref, yref
#                     if self._dim == 3:
#                         annotation.update({"x": y, "y": x})
#                         if entry.composition.is_element:
#                             z = 0.9 * self._min_energy  # place label 10% above base
#    
#                 annotation.update({"z": z})
#   
#             annotations_list.append(annotation)
#   
#         # extra point ensures equilateral triangular scaling is displayed
# #         if self._dim == 3:
# #             annotations_list.append(dict(x=1, y=1, z=0, opacity=0, text=""))
#   
#         return annotations_list
    
    def _create_plotly_figure_layout(self, label_stable=True):
        """
        Creates layout for plotly phase diagram figure and updates with
        figure annotations.

        :return: Dictionary with Plotly figure layout settings.
        """
        annotations_list = None
        layout = dict()

        if label_stable:
            if isinstance(label_stable, dict):
                annotations_list = self._create_plotly_element_annotations_change_name(label_stable)
            else:
                annotations_list = self._create_plotly_element_annotations()

        if self._dim == 2:
            layout = plotly_layouts["default_binary_layout"].copy()
            layout["annotations"] = annotations_list
        elif self._dim == 3:
            layout = plotly_layouts["default_ternary_layout"].copy()
            layout["scene"].update({"annotations": annotations_list})
        elif self._dim == 4:
            layout = plotly_layouts["default_quaternary_layout"].copy()
            layout["scene"].update({"annotations": annotations_list})

        return layout
    
    def _create_plotly_element_annotations_change_name(self,namedict):
        """
        Creates terminal element annotations for Plotly phase diagrams.

        Returns:
            list of annotation dicts.
        """
        annotations_list = []
        x, y, z = None, None, None

        for coords, entry in self.pd_plot_data[1].items():
            if not entry.composition.is_element:
                continue

            x, y = coords[0], coords[1]

            if self._dim == 3:
                z = self._pd.get_form_energy_per_atom(entry)
            elif self._dim == 4:
                z = coords[2]

            if entry.composition.is_element:
                clean_formula = str(entry.composition.elements[0])
                clean_formula = namedict[clean_formula]
                if hasattr(entry, "original_entry"):
                    orig_comp = entry.original_entry.composition
                    clean_formula = htmlify(orig_comp.reduced_formula)
                    clean_formula = namedict[clean_formula]
                font_dict = {"color": "#000000", "size": 24.0}
                opacity = 1.0

            annotation = plotly_layouts["default_annotation_layout"].copy()
            annotation.update(
                {
                    "x": x,
                    "y": y,
                    "font": font_dict,
                    "text": clean_formula,
                    "opacity": opacity,
                }
            )

            if self._dim in (3, 4):
                for d in ["xref", "yref"]:
                    annotation.pop(d)  # Scatter3d cannot contain xref, yref
                    if self._dim == 3:
                        annotation.update({"x": y, "y": x})
                        if entry.composition.is_element:
                            z = 0.9 * self._min_energy  # place label 10% above base

                annotation.update({"z": z})

            annotations_list.append(annotation)

        # extra point ensures equilateral triangular scaling is displayed
        if self._dim == 3:
            annotations_list.append(dict(x=1, y=1, z=0, opacity=0, text=""))

        return annotations_list
    
    def create_reaction_plane(self):
        '''xx yy zz are elemental mu at the intercepts (the order is the elements order)'''
        # BaN2 
        aa = (0.3333333333333333, 0.5773502691896257,-0.7648113232533321)
        # MnN 
        bb = (0.75, 0.4330127018922193,-0.4743314211899978)
        para = 1.5
        coords = np.array([[0.3333333333333333, 0.5773502691896257,0],[0.3333333333333333, 0.5773502691896257,self._min_energy*para],
                           [0.75, 0.4330127018922193,0],[0.75, 0.4330127018922193,self._min_energy*para]])
        print(coords[:,0])
        data1 = [go.Scatter3d(
            x=[aa[1],bb[1]], y=[aa[0],bb[0]], z=[aa[2],bb[2]],
                marker=dict(
                size=8,
                color="darkblue",
                colorscale='Viridis',
            ),
            line=dict(
                color='red',
                width=5
            ))]
        
        facets = [0,1,2,3]
        # i, j and k give the vertices of triangles
        # here we represent the 2 triangles of the rectangle
        data1.append(go.Mesh3d(
            x=list(coords[:, 1]),
            y=list(coords[:, 0]),
            z=list(coords[:, 2]),
            i=[facets[1],facets[1]],
            j=[facets[0],facets[2]],
            k=[facets[2],facets[3]],
            opacity=0.8,
            color = "lightblue",
            hoverinfo="none",
            lighting=dict(diffuse=0.0, ambient=1.0),
            name="Convex Hull (shading)",
            flatshading=True,
            showlegend=True,
        ))
        return data1
    def create_reaction_plane2(self):
        '''xx yy zz are elemental mu at the intercepts (the order is the elements order)'''
        # BaN2 
        aa = (0.3333333333333333, 0.5773502691896257,-0.7648113232533321)
        '''plot a series planes with AxB1-x'''
        para = 1.5
        data1 = []
        w = 5
        for ii in [0.5,0.33]:
            if ii == 0.5:
                c = ["#cc33ff","#e699ff"]
                c = ["#4d94ff","#b3d1ff"]
#                 w = 10
            else:
                c = ["#4d94ff","#b3d1ff"]
            coords = np.array([np.append(triangular_coord((ii,0)),0),np.append(triangular_coord((ii,0)),self._min_energy*para),
                               np.append(triangular_coord((0,1)),0),np.append(triangular_coord((0,1)),self._min_energy*para)])
            facets = [0,1,2,3]
            print(coords)
            print(np.append(triangular_coord((0,1)),0))
            # i, j and k give the vertices of triangles
            # here we represent the 2 triangles of the rectangle
            for aa,bb in zip([0,0,2,1],[1,2,3,3]):
                data1.append(go.Scatter3d(
                x=[coords[aa][1],coords[bb][1]], y=[coords[aa][0],coords[bb][0]], z=[coords[aa][2],coords[bb][2]],
                    marker=dict(
                    size=6,
                    opacity=0.5,
                    color=c[0],
                    colorscale='Viridis',
                ),
                line=dict(
                    color=c[0],
                    width=w
                )))
            if ii == 0.8:
                data1.append(go.Mesh3d(
                    x=list(coords[:, 1]),
                    y=list(coords[:, 0]),
                    z=list(coords[:, 2]),
                    i=[facets[1],facets[1]],
                    j=[facets[0],facets[2]],
                    k=[facets[2],facets[3]],
                    opacity=0.2,
                    color = c[1],
                    hoverinfo="none",
                    lighting=dict(diffuse=0.0, ambient=1.0),
                    name="Convex Hull (shading)",
                    flatshading=True,
                    showlegend=True,
                ))
        return data1

    def create_reaction_plane3(self,compositions,marksize=10,linewidth=8,
                               color = None, para = 1.5, alpha = 0.5):
        '''input compositions of two entries you want to link in a rectangle'''

        '''plot a series planes with AxB1-x'''
        
        data1 = []
        comp_vers = [[comp.get_atomic_fraction(self._pd.elements[1]),
                      comp.get_atomic_fraction(self._pd.elements[2])] for comp in compositions]
        ii = comp_vers[0]
        jj = comp_vers[1]
        c = ["#cc33ff","#e699ff"]
#                 w = 10
#         else:
#             c = ["#4d94ff","#b3d1ff"]
        coords = np.array([np.append(triangular_coord((ii[0],ii[1])),0),np.append(triangular_coord((ii[0],ii[1])),self._min_energy*para),
                           np.append(triangular_coord((jj[0],jj[1])),0),np.append(triangular_coord((jj[0],jj[1])),self._min_energy*para)])
        facets = [0,1,2,3]
        print(coords)
        print(np.append(triangular_coord((0,1)),0))
        # i, j and k give the vertices of triangles
        # here we represent the 2 triangles of the rectangle
        for aa,bb in zip([0,0,2,1],[1,2,3,3]):
            data1.append(go.Scatter3d(
            x=[coords[aa][1],coords[bb][1]], y=[coords[aa][0],coords[bb][0]], z=[coords[aa][2],coords[bb][2]],
                marker=dict(
                size=0, # if irpd too much, without boundary marker might be better
                opacity=0.5,
                color=c[0] if color is None else color[0],
                colorscale='Viridis',
            ),
            line=dict(
                color=c[0] if color is None else color[0],
                width=linewidth
            )))

        data1.append(go.Mesh3d(
            x=list(coords[:, 1]),
            y=list(coords[:, 0]),
            z=list(coords[:, 2]),
            i=[facets[1],facets[1]],
            j=[facets[0],facets[2]],
            k=[facets[2],facets[3]],
            opacity=alpha,
            color = c[1] if color is None else color[1],
            hoverinfo="none",
            lighting=dict(diffuse=0.0, ambient=1.0),
            name="Convex Hull (shading)",
            flatshading=True,
            showlegend=True,
        ))
        return data1
    
    def get_irpd(self,comp1=Composition("N2"),comp2=Composition("BaMn4"),marksize=6,linewidth=8,
                 color = None):
        c = ["#cc33ff","#e699ff"]
#         print(comp1.reduced_formula)
#         print(comp2.reduced_formula)

#         pd = PhaseDiagram(self._pd.original_entries)
#         pd = PhaseDiagram(self._pd.stable_entries)
        pd = self._pd
        cricomps = pd.get_critical_compositions(comp1,comp2)
        print()
        vertices = []
        for comp in cricomps:
            formE = (pd.get_hull_energy(comp)- sum([comp[el] * pd.el_refs[el].energy_per_atom for el in comp.elements]))/comp.num_atoms
            ver = np.append(triangular_coord([comp.get_atomic_fraction(pd.elements[1]),
                              comp.get_atomic_fraction(pd.elements[2])]),formE)
            print(comp)
#             print(comp.get_atomic_fraction(pd.elements[1]))
            print(ver)
            vertices.append(ver.tolist())
        vertices = np.array(vertices)
        print(vertices)
        print(vertices[:,0])
        data1 = []
        for i in range(0,len(vertices)-1):
            cc = c[0]
#             if i==2:
#                 cc="#4d94ff"
            print(vertices[:,0][i:i+2])
            data1.append(go.Scatter3d(
                x=vertices[:,1][i:i+2], y=vertices[:,0][i:i+2], z=vertices[:,2][i:i+2],
                    marker=dict(
                    size=marksize,
                    color=cc if color is None else color[0],
                    colorscale='Viridis',
                ),
                line=dict(
                    color=cc if color is None else color[0],
#                     color = "black",
                    width=linewidth)
                ))
        return data1
    
    def add_scatter(self, scatters, colors = None, marksize = 8, symbol = "circle"):
        data = []
        if colors is None:
            colors = ["red"] * len(scatters)
        for scatter, c in zip(scatters, colors):
            for coords, entry in self.pd_plot_data[1].items():
    
                if entry.composition == scatter.composition:
                    x_coord = coords[0]
                    y_coord = coords[1]
                    z_coord = round(self._pd.get_form_energy_per_atom(scatter), 3)
                    break
            data.append(
                go.Scatter3d(
                    mode='markers',
                    x=[y_coord],
                    y=[x_coord],
                    z=[z_coord],
                    marker=dict(
                        color= c,
                        size=marksize,
                        symbol = symbol
                    ),
                    showlegend=False
                )
            )
        return data

    def create_tangent_plane(self,entryname = "Ge2N2O",marksize=8,linewidth=8):

        '''xx yy zz are elemental mu at the intercepts (the order is the elements order)'''
        CPentries = trans_PD_to_ChemPot_entries(self._pd.stable_entries, 
                                                [str(el) for el in self._pd.elements])
        cp=EquilLine(CPentries, [str(el) for el in self._pd.elements])
        for entry, vertices in cp._stable_domain_vertices.items():
            print(entry.name)
            if entry.name == entryname:
                tarentry = entry
                # xx, yy, zz can be any point in cp diagram, does not have to be average
                xx,yy,zz = np.average(vertices, axis=0)
        print(xx,yy,zz)
        
        data1 = []
        # the plane passing through three points of coords has to pass the target material in the convex hull
        coords = np.array([[0,0,xx],[1,0,yy],[0.5,0.8660254037844386,zz]])
        plotline = False
        if plotline:
            Ba = coords[0]
            Mn = coords[1]
            N = coords[2]
            point1 = (Ba-N)*(1-tarentry.composition.get_atomic_fraction(self._pd.elements[-1]))+N
            point2 = (Mn-N)*(1-tarentry.composition.get_atomic_fraction(self._pd.elements[-1]))+N
            
            data1.append(go.Scatter3d(
                x=[point1[1],point2[1]], y=[point1[0],point2[0]], z=[point1[2],point2[2]],
                    marker=dict(
                    size=marksize,
                    opacity=0.5,
                    color="#4d94ff",
                    colorscale='Viridis',
                ),
                line=dict(
                    dash='dash',
                    color="#4d94ff",
                    width=linewidth
                )))
        print(coords[:,0])
        facets = [0,1,2]
        for aa,bb in zip([0,1,2],[1,2,0]):
            data1.append(go.Scatter3d(
            x=[coords[aa][1],coords[bb][1]], y=[coords[aa][0],coords[bb][0]], z=[coords[aa][2],coords[bb][2]],
                marker=dict(
                size=marksize,
                opacity=0.5,
                color="#4d94ff",
                colorscale='Viridis',
            ),
            line=dict(
                color="#4d94ff",
                width=linewidth
            )))
        data1.append(go.Mesh3d(
            x=list(coords[:, 1]),
            y=list(coords[:, 0]),
            z=list(coords[:, 2]),
            i=[facets[1]],
            j=[facets[0]],
            k=[facets[2]],
            opacity=0.2,
            color = "#4d94ff",
            hoverinfo="none",
            lighting=dict(diffuse=0.0, ambient=1.0),
            name="Convex Hull (shading)",
            flatshading=True,
            showlegend=True,
        ))
        return data1
        

    def _create_plotly_ternary_hull(self):
        """
        Creates shaded mesh plot for coloring the ternary hull by formation energy.

        :return: go.Mesh3d plot
        """
        facets = np.array(self._pd.facets)
        coords = np.array(
            [
                triangular_coord(c)
                for c in zip(self._pd.qhull_data[:-1, 0], self._pd.qhull_data[:-1, 1])
            ]
        )
#         for c in zip(self._pd.qhull_data[:-1, 0], self._pd.qhull_data[:-1, 1]):
#             print("haha",c, triangular_coord(c))
        names = np.array(
            [e.name for e in self._pd.qhull_entries]
        )
        
        energies = np.array(
            [self._pd.get_form_energy_per_atom(e) for e in self._pd.qhull_entries]
        )

        return go.Mesh3d(
            x=list(coords[:, 1]),
            y=list(coords[:, 0]),
            z=list(energies),
            i=list(facets[:, 1]),
            j=list(facets[:, 0]),
            k=list(facets[:, 2]),
            opacity=0.8,
            intensity=list(energies),
            colorscale=plotly_layouts["stable_colorscale"],
            # colorscale = [[
            #                   0.0,
            #                   "#908e83"
            #                 ],
            #                 [
            #                   0.5,
            #                   "#e9e8e4"
            #                 ],
            #                 [
            #                   1.0,
            #                   "#ffffff"
            #                 ]],
            colorbar=dict(title="Formation energy<br>(eV/atom)", x=0.9, len=0.75),
            hoverinfo="none",
            lighting=dict(diffuse=0.0, ambient=1.0),
            name="Convex Hull (shading)",
            flatshading=True,
            showlegend=True,
        )

    def create_ternary_in_quanternary_hull(self, terminal_comp = [["NiO", "CoO","Cr2O3"]]):
        """
        Creates shaded mesh plot for coloring the ternary hull by formation energy.

        :return: go.Mesh3d plot
        """
        
        terminals = [[Composition(i) for i in tc] for tc in terminal_comp]
        entries = self._pd._stable_entries
        data = []
        for terminal in terminals:
            cpd = CompoundPhaseDiagram(entries, terminal)
            facets = np.array(cpd.facets)
    
            energies = []
            names = []
            '''qhull entries' sequence is as same as facets, not all_entries'''
            for e in cpd.qhull_entries:
                names.append(e.name)
#                 energies.append(cpd.get_form_energy_per_atom(e))
                energies.append(self._pd.get_form_energy_per_atom(e.original_entry))
            coords = []
            for name in names:
                for coord, entry in self.pd_plot_data[1].items():
                    if name == entry.name:
                        print(entry.name, coord)
                        coords.append(coord)
    
            coords  = np.array(coords)

            data.append(go.Mesh3d(
                x=list(coords[:, 0]),
                y=list(coords[:, 1]),
                z=list(coords[:, 2]),
                i=list(facets[:, 0]),
                j=list(facets[:, 1]),
                k=list(facets[:, 2]),
                opacity=0.8,
                intensity=list(energies),
#                 colorscale=plotly_layouts["stable_colorscale"],
                colorscale=[
                                [
                                  0.0,
                                  "#908e83"
                                ],
                                [
                                  0.5,
                                  "#e9e8e4"
                                ],
                                [
                                  1.0,
                                  "#ffffff"
                                ]
                              ],
                colorbar=dict(title="Formation energy<br>(eV/atom)", x=0.9, len=0.75),
                hoverinfo="none",
                lighting=dict(diffuse=0.0, ambient=1.0),
                name="Convex Hull (shading)",
                flatshading=True,
                showlegend=True,
                ))
        return data

    def create_ternary_in_quanternary_hull_triangle_colormap(
            self, step = 0.001, opacity = 0.04):
        """
        Creates shaded mesh plot for coloring the ternary hull by formation energy.

        :return: go.Mesh3d plot
        """
        
        data = []
        # Define the coordinates of the three vertices of the triangle
        vertex1 = np.array([0.0, 0.0, 0.0])
        vertex2 = np.array([1.0, 0.0, 0.0])
        vertex3 = np.array([0.5, math.sqrt(3) / 2, 0.0])
        
        # Define the step size for generating the grid
        step = step
        
        # Create a grid
        x = np.arange(vertex1[0], vertex2[0] + step, step)
        y = np.arange(vertex1[1], vertex3[1] + step, step)
        
        # Initialize the list of point coordinates and colors
        points = []
        colors = []
        
        # Iterate through each point in the grid
        for xi in x:
            for yi in y:
                # Calculate if the point is inside the triangle
                v0 = vertex3 - vertex1
                v1 = vertex2 - vertex1
                v2 = np.array([xi, yi, 0.0]) - vertex1
        
                dot00 = np.dot(v0, v0)
                dot01 = np.dot(v0, v1)
                dot02 = np.dot(v0, v2)
                dot11 = np.dot(v1, v1)
                dot12 = np.dot(v1, v2)
        
                # Calculate barycentric coordinates
                inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
                # If the point is inside the triangle, add it to the list of point coordinates
                if (u >= 0) and (v >= 0) and (u + v <= 1):
                    points.append([xi, yi, 0.0])
                    # The color scheme can be changed as needed

                    colors.append(get_color_tri([xi, yi]))

        # Convert the lists of point coordinates and colors to NumPy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Create the Scatter3d object
        trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=colors,
                opacity=opacity
            )
        )
        data.append(trace)
        return data
    def _create_plotly_lines(self):
        """
        Creates Plotly scatter (line) plots for all phase diagram facets.

        :return: go.Scatter (or go.Scatter3d) plot
        """
        line_plot = None
        x, y, z, energies = [], [], [], []

        for line in self.pd_plot_data[0]:
            print(line)
            x.extend(list(line[0]) + [None])
            y.extend(list(line[1]) + [None])

            if self._dim == 3:
                z.extend(
                    [
                        self._pd.get_form_energy_per_atom(self.pd_plot_data[1][coord])
                        for coord in zip(line[0], line[1])
                    ]
                    + [None]
                )

            elif self._dim == 4:
                energies.extend(
                    [
                        self._pd.get_form_energy_per_atom(self.pd_plot_data[1][coord])
                        for coord in zip(line[0], line[1], line[2])
                    ]
                    + [None]
                )
                for coord in zip(line[0], line[1], line[2]):
                    print(coord)
                    print(self.pd_plot_data[1][coord].name)

                z.extend(list(line[2]) + [None])

        plot_args = dict(
            mode="lines",
            name = "lines",
            hoverinfo="none",
            line={"color": "rgba(0,0,0,1.0)", "width": 3.0},
            showlegend=True,
        )

        if self._dim == 2:
            line_plot = go.Scatter(x=x, y=y, **plot_args)
        elif self._dim == 3:
            line_plot = go.Scatter3d(x=y, y=x, z=z, **plot_args)
        elif self._dim == 4:
            line_plot = go.Scatter3d(x=x, y=y, z=z, **plot_args)

        return line_plot


    def _create_plotly_ternary_support_lines(self,para = 1.5):
        """
        Creates support lines which aid in seeing the ternary hull in three
        dimensions.

        :return: go.Scatter3d plot of support lines for ternary phase diagram.
        """
        stable_entry_coords = dict(map(reversed, self.pd_plot_data[1].items()))
        for e in stable_entry_coords:
            print(e.name,stable_entry_coords[e])
        print()
        elem_coords = [stable_entry_coords[e] for e in self._pd.el_refs.values()]
        print(elem_coords)

        # add top and bottom triangle guidelines
        x, y, z = [], [], []
        for line in itertools.combinations(elem_coords, 2):
            x.extend([line[0][0], line[1][0], None] * 2)
            y.extend([line[0][1], line[1][1], None] * 2)
            z.extend([0, 0, None, self._min_energy*para, self._min_energy*para, None])

        # add vertical guidelines
        for elem in elem_coords:
            x.extend([elem[0], elem[0], None])
            y.extend([elem[1], elem[1], None])
            z.extend([0, self._min_energy*para, None])

        return go.Scatter3d(
            x=list(y),
            y=list(x),
            z=list(z),
            mode="lines",
            hoverinfo="none",
            line=dict(color="rgba (0, 0, 0, 0.4)", dash="solid", width=5.0),
            showlegend=False,
        )
    def _create_plotly_ternary_support_lines2(self,para = 1.5):
        """
        Creates support lines which aid in seeing the ternary hull in three
        dimensions.

        :return: go.Scatter3d plot of support lines for ternary phase diagram.
        """
        stable_entry_coords = dict(map(reversed, self.pd_plot_data[1].items()))
        for e in stable_entry_coords:
            print(e.name,stable_entry_coords[e])
        print()
        elem_coords = [stable_entry_coords[e] for e in self._pd.el_refs.values()]
        print(elem_coords)

        # add top and bottom triangle guidelines
        x, y, z = [], [], []
        for line in itertools.combinations(elem_coords, 2):
            x.extend([line[0][0], line[1][0], None]*2)
            y.extend([line[0][1], line[1][1], None]*2)
            z.extend([0, 0, None,self._min_energy*1.5, self._min_energy*1.5, None])

        # add vertical guidelines
        for elem in elem_coords:
            x.extend([elem[0], elem[0], None])
            y.extend([elem[1], elem[1], None])
            z.extend([0, self._min_energy*1.5, None])

        data =[go.Scatter3d(
            x=list(y),
            y=list(x),
            z=list(z),
            mode="lines",
            hoverinfo="none",
            line=dict(color="rgba (0, 0, 0, 0.5)", dash="solid", width=5.0),
            showlegend=False,
        )]
        
        x, y, z = [], [], []
        for line in itertools.combinations(elem_coords, 2):
            x.extend([line[0][0], line[1][0], None])
            y.extend([line[0][1], line[1][1], None])
            z.extend([self._min_energy*para, self._min_energy*para, None])
        data += [go.Scatter3d(
            x=list(y),
            y=list(x),
            z=list(z),
            mode="lines",
            hoverinfo="none",
            line=dict(color="rgba (0, 0, 0, 0)", dash="solid", width=5.0),
            showlegend=False,
        )]
        return data
        
    def _create_plotly_markers(self, label_uncertainties=False,marksize=5, colorblue = False):
        """
        Creates stable and unstable marker plots for overlaying on the phase diagram.

        :return: Tuple of Plotly go.Scatter (or go.Scatter3d) objects in order: (
            stable markers, unstable markers)
        """

        def get_marker_props(coords, entries, stable=True):
            """ Method for getting marker locations, hovertext, and error bars
            from pd_plot_data"""
            x, y, z, texts, energies, uncertainties = [], [], [], [], [], []
            colors = []
            colordict = {'Mn2N': [1.0, 0.0, 0.16, 1.0], 'N2': [1.0, 0.325384207737149, 0.0, 1.0], 'MnN': [1.0, 0.81293057763646, 0.0, 1.0], 'Mn4N': [0.6995230524642289, 1.0, 0.0, 1.0], 'Mn': [0.19077901430842603, 1.0, 0.0, 1.0], 'BaN6': [0.0, 1.0, 0.29517183217372983, 1.0], 'BaMnN2': [0.0, 1.0, 0.7800969850305717, 1.0], 'BaN2': [0.0, 0.7320971867007668, 1.0, 1.0], 'Ba3MnN3': [0.0, 0.22058823529411742, 1.0, 1.0], 'Ba2N': [0.2696078431372551, 0.0, 1.0, 1.0], 'Ba3N': [0.7598039215686277, 0.0, 1.0, 1.0], 'Ba': [1.0, 0.0, 0.75, 1.0]}
            colordict = {'Li6ZnO4': [1.0, 0.0, 0.16, 1.0], 'Li2O': [1.0, 0.2405935347111818, 0.0, 1.0], 'ZnO': [1.0, 0.6645468998410176, 0.0, 1.0], 'Li4P2O7': [0.9114997350291467, 1.0, 0.0, 1.0], 'Li4Zn(PO4)2': [0.5087440381558028, 1.0, 0.0, 1.0], 'ZnP4O11': [0.08479067302596743, 1.0, 0.0, 1.0], 'Zn(PO3)2': [0.0, 1.0, 0.33733923676997696, 1.0], 'Zn3(PO4)2': [0.0, 1.0, 0.7379295804343246, 1.0], 'Zn2P2O7': [0.0, 0.8386615515771522, 1.0, 1.0], 'LiZnPO4': [0.0, 0.412404092071611, 1.0, 1.0], 'LiPO3': [0.0, 0.007459505541347444, 1.0, 1.0], 'P2O5': [0.4187979539641946, 0.0, 1.0, 1.0], 'Li3PO4': [0.8450554134697361, 0.0, 1.0, 1.0], 'Li10Zn4O9': [1.0, 0.0, 0.75, 1.0]}
            colordict = {'Co3Ni': [0.75, 0.0, 0.25, 1.0], 'CrNi2': [-0.0, 0.3333, 0.6667, 1.0], 'Ni': [0.0, 0.0, 1.0, 1.0], 'Cr': [0.0, 1.0, 0.0, 1.0], 'Co': [1.0, 0.0, 0.0, 1.0], 'Cr2O3': [0.0, 1.0, 0.0, 0.7407], 'Cr2CoO4': [0.3333, 0.6667, 0.0, 0.6032], 'CoO': [1.0, 0.0, 0.0, 0.5115], 'NiO': [0.0, 0.0, 1.0, 0.4894], 'Co3O4': [1.0, 0.0, 0.0, 0.2806], 'CrNiO4': [0.0, 0.5, 0.5, 0.18], 'Co(NiO2)2': [0.3333, 0.0, 0.6667, 0.1624], 'CrO2': [0.0, 1.0, 0.0, 0.1589], 'CoNiO3': [0.5, 0.0, 0.5, 0.1587], 'CoO2': [1.0, 0.0, 0.0, 0.153], 'CrCoO4': [0.5, 0.5, 0.0, 0.1367], 'Cr5O12': [0.0, 1.0, 0.0, 0.1193], 'Ni3O4': [0.0, 0.0, 1.0, 0.0865]}


            
            sizes = []
            for coord, entry in zip(coords, entries):
                sizes.append(marksize)
                energy = round(self._pd.get_form_energy_per_atom(entry), 4)
                if entry.name in colordict:
                    colors.append(colordict[entry.name])
                else:
                    colors.append("black")
                entry_id = getattr(entry, "entry_id", "no ID")
                comp = entry.composition

                if hasattr(entry, "original_entry"):
                    comp = entry.original_entry.composition

                formula = comp.reduced_formula
                clean_formula = htmlify(formula)
                label = f"{clean_formula} ({entry_id}) <br> " f"{energy} eV/atom"

                if not stable:

                    e_above_hull = round(self._pd.get_e_above_hull(entry), 3)
                    if e_above_hull > self.show_unstable:
                        continue
                    label += f" (+{e_above_hull} eV/atom)"
                    energies.append(e_above_hull)
                else:
                    uncertainty = 0
                    if (
                        hasattr(entry, "correction_uncertainty_per_atom")
                        and label_uncertainties
                    ):
                        uncertainty = round(entry.correction_uncertainty_per_atom, 4)
                        label += f"<br> (Error: +/- {uncertainty} eV/atom)"

                    uncertainties.append(uncertainty)
                    energies.append(energy)

                texts.append(label)

                x.append(coord[0])
                y.append(coord[1])

                if self._dim == 3:
                    z.append(energy)
                elif self._dim == 4:
                    z.append(coord[2])

            return {
                "x": x,
                "y": y,
                "z": z,
                "size":sizes,
                "texts": texts,
                "energies": energies,
                "uncertainties": uncertainties,
                "colors":colors
            }

        stable_coords, stable_entries = (
            self.pd_plot_data[1].keys(),
            self.pd_plot_data[1].values(),
        )
        unstable_entries, unstable_coords = (
            self.pd_plot_data[2].keys(),
            self.pd_plot_data[2].values(),
        )

        stable_props = get_marker_props(stable_coords, stable_entries)

        unstable_props = get_marker_props(
            unstable_coords, unstable_entries, stable=False
        )

        stable_markers, unstable_markers = dict(), dict()

        if self._dim == 2:
            stable_markers = plotly_layouts["default_binary_marker_settings"].copy()
            stable_markers.update(
                dict(
                    x=list(stable_props["x"]),
                    y=list(stable_props["y"]),
                    name="Stable",
                    marker=dict(
                        color="darkgreen", size=11, line=dict(color="black", width=2)
                    ),
                    opacity=0.9,
                    hovertext=stable_props["texts"],
                    error_y=dict(
                        array=list(stable_props["uncertainties"]),
                        type="data",
                        color="gray",
                        thickness=2.5,
                        width=5,
                    ),
                )
            )

            unstable_markers = plotly_layouts["default_binary_marker_settings"].copy()
            unstable_markers.update(
                dict(
                    x=list(unstable_props["x"]),
                    y=list(unstable_props["y"]),
                    name="Above Hull",
                    marker=dict(
                        color=unstable_props["energies"],
                        colorscale=plotly_layouts["unstable_colorscale"],
                        size=15,
                        symbol="diamond",
                    ),
                    hovertext=unstable_props["texts"],
                )
            )

        elif self._dim == 3:
            stable_markers = plotly_layouts["default_ternary_marker_settings"].copy()
            stable_markers.update(
                dict(
                    x=list(stable_props["y"]),
                    y=list(stable_props["x"]),
                    z=list(stable_props["z"]),
#                     z=[0 for ii in range(len(list(stable_props["z"])))],
                    name="Stable",
                    marker=dict(
                        color = stable_props["colors"] if not colorblue else "#005ce6",
                        size=stable_props["size"],
#                         size=marksize,
                        opacity=0.6,
                        line=dict(width=3,color="black"),
                    ),
#                     marker_symbol="square",
                    hovertext=stable_props["texts"],
                    error_z=dict(
                        array=list(stable_props["uncertainties"]),
                        type="data",
                        color="darkgray",
                        width=10,
                        thickness=5,
                    ),
                )
            )

            unstable_markers = plotly_layouts["default_ternary_marker_settings"].copy()
            unstable_markers.update(
                dict(
                    x=unstable_props["y"],
                    y=unstable_props["x"],
                    z=unstable_props["z"],
                    name="Above Hull",
                    marker=dict(
                        color=unstable_props["energies"],
                        colorscale=plotly_layouts["unstable_colorscale"],
                        size=6,
                        symbol="diamond",
                        colorbar=dict(
                            title="Energy Above Hull<br>(eV/atom)", x=0.05, len=0.75
                        ),
                    ),
                    hovertext=unstable_props["texts"],
                )
            )

        elif self._dim == 4:
            stable_markers = plotly_layouts["default_quaternary_marker_settings"].copy()
            stable_markers.update(
                dict(
                    x=stable_props["x"],
                    y=stable_props["y"],
                    z=stable_props["z"],
                    name="Stable",
                    marker=dict(
                        color=stable_props["colors"] if not colorblue else "#005ce6",
#                         color=stable_props["energies"],
#                         colorscale=plotly_layouts["stable_markers_colorscale"],
                        size=8,
                        opacity=1,
                    ),
                    hovertext=stable_props["texts"],
                )
            )

            unstable_markers = plotly_layouts[
                "default_quaternary_marker_settings"
            ].copy()
            unstable_markers.update(
                dict(
                    x=unstable_props["x"],
                    y=unstable_props["y"],
                    z=unstable_props["z"],
                    name="Above Hull",
                    marker=dict(
                        color=unstable_props["energies"],
                        colorscale=plotly_layouts["unstable_colorscale"],
                        size=5,
                        symbol="diamond",
                        colorbar=dict(
                            title="Energy Above Hull<br>(eV/atom)", x=0.05, len=0.75
                        ),
                    ),
                    hovertext=unstable_props["texts"],
                    visible="legendonly",
                )
            )

        stable_marker_plot = (
            go.Scatter(**stable_markers)
            if self._dim == 2
            else go.Scatter3d(**stable_markers)
        )
        unstable_marker_plot = (
            go.Scatter(**unstable_markers)
            if self._dim == 2
            else go.Scatter3d(**unstable_markers)
        )

        return stable_marker_plot, unstable_marker_plot
        









