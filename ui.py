# -*- coding: utf-8 -*-
# Copyright 2023 Ubiratan S. Freitas


# This file is part of Lattesh.
# 
# Lattesh is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
# 
# Lattesh is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
# more details.
# 
# You should have received a copy of the GNU General Public License along with 
# Lattice Tools. If not, see <https://www.gnu.org/licenses/>. 


from bpy.types import Panel


from .operators import have_scipy


# Test functions for polling
def is_mode_object(context):
    return context.mode == 'OBJECT'

def is_active_object_mesh(context):
    return context.active_object is not None and context.active_object.type == 'MESH'


class View3DLatticePanel:
    bl_category = "Lattices"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'


class VIEW3D_PT_lattices_generate(View3DLatticePanel, Panel):
    bl_label = "Generate lattice skeletons"
    bl_idname = "VIEW3D_PT_lattices_generate"

    def draw(self, context):
        layout = self.layout

        has_target = context.active_object is not None \
                    and context.active_object.type == 'MESH' \
                    and context.active_object.select_get()

        lattice_mesh = context.scene.lattice_mesh

        layout.label(text="Regular cells")
        row = layout.row(align=True)
        row.label(text="Cell size")
        row.prop(lattice_mesh, "cellsize", text="")
        row = layout.row(align=True)
        row.label(text="Cell type")
        row.prop(lattice_mesh, "celltype", text="")

        box = layout.box()
        box.operator("object.create_meshlattice")
        box.label(text="Number of cells")
        row = box.row()
        row.prop(lattice_mesh, "nelems", text="")

        box = layout.box()
        box.operator("object.fill_meshlattice")
        box.prop(lattice_mesh, "move_dist")
        row = box.row()
        row.prop(lattice_mesh, "filter_angle")
        row.prop(lattice_mesh, "min_angle", text="")
        box.prop(lattice_mesh, "connect_boundary", text="Connect boundary")
        box.prop(lattice_mesh, "create_faces")
        box.operator("object.filter_lattice")
        box.enabled = has_target


#        box = layout.box()
#        box.operator("object.fill_voronoilattice")
#        box.prop(lattice_mesh, "charac_length", text="Char. length")
#        box.enabled = has_target


class VIEW3D_PT_lattice_Ico(View3DLatticePanel, Panel):
    bl_label = "Icosphere fill"
    bl_idname = "VIEW3D_PT_lattice_Ico"

    @classmethod
    def poll(cls, context):
        return is_mode_object(context) and is_active_object_mesh(context) and have_scipy

    def draw(self, context):
        lattice_mesh = context.scene.lattice_mesh
        depsgraph = context.evaluated_depsgraph_get()
        has_radius = 'radius' in context.active_object.evaluated_get(depsgraph).data.attributes
        layout = self.layout
        radius_row = layout.row(align=True)
        radius_row.label(text="Strut radius")
        radius_row.prop(lattice_mesh, "radius", text="")
        if has_radius and lattice_mesh.variable_radius:
            radius_row.enabled = False
        var_row = layout.row(align=True)
        var_row.label(text="Variable radius")
        var_row.prop(lattice_mesh, "variable_radius", text="")
        var_row.enabled = has_radius
        sub_row = layout.row(align=True)
        sub_row.label(text="Icosphere subdivisions")
        sub_row.prop(lattice_mesh, "icosub", text="")
        sub_row = layout.row(align=False)
        sub_row.prop(lattice_mesh, "add_index",text="Add index")
        sub_row.prop(lattice_mesh, "add_dia_mod")

        layout.operator("object.icolattice")


class VIEW3D_PT_lattice_metaball(View3DLatticePanel, Panel):
    bl_label = "Metaball fill"
    bl_idname = "VIEW3D_PT_lattice_metaball"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return is_mode_object(context) and is_active_object_mesh(context)

    def draw(self, context):
        lattice_mesh = context.scene.lattice_mesh

        layout = self.layout
        row = layout.row(align=True)
        row.label(text="Strut radius")
        row.prop(lattice_mesh, "radius", text="")
        row = layout.row(align=True)
        row.label(text="Radius ratio")
        row.prop(lattice_mesh, "r_ratio", text="")

        layout.operator("object.metalattice")

