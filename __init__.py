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



bl_info = {
    "name": "Lattesh - Lattice Tools",
    "author": "Ubiratan S. Freitas",
    "blender": (2, 80, 0),
    "description": "Tools for generating lattice structures",
    "category": "Add Mesh",
}






if "bpy" in locals():
    import importlib
    importlib.reload(operators)
    importlib.reload(ui)
else:
    import bpy
    import math
    from bpy.types import PropertyGroup
    from bpy.props import (
        BoolProperty,
        FloatProperty,
        EnumProperty,
        IntVectorProperty,
        PointerProperty,
        IntProperty
    )

    from . import (
        operators,
        ui,
    )


cell_items = operators.cell_items

class SceneProperties(PropertyGroup):
    r_ratio: FloatProperty(
        name="R_ratio",
        description="Ratio of actual radius to metaball radius",
        default=0.65, 
        min=0.01,
        max=0.99,
    )

    radius: FloatProperty(
        name="Radius",
        description="Actual radius of resulting mesh",
        default=0.5, 
        min=1.0e-6,
        max=20.0,
        unit='LENGTH',
    )

    cellsize: FloatProperty(
        name="Cell size",
        description="Cell size",
        default=10, 
        min=1.0e-2,
        unit='LENGTH',
    )

    nelems: IntVectorProperty(
        name="Nr. cells",
        description="Number of cells in each direction",
        default=(5, 5, 3), 
        min=1,
        size=3,
    )

    celltype: EnumProperty(
        items=cell_items,
        name="Cell type",
        description="Cell type",
        default=cell_items[0][0],
    )

    connect_boundary: BoolProperty(
        name="Connect Boundary",
        description="When filling a volume, connect the loose struts on the boundary if the cell file supports it",
        default=True,
    )

    add_index: BoolProperty(
        name="Add index",
        description="Add an index to the original skeleton vertex to each vertex on the created mesh",
        default=True,
    )

    variable_radius: BoolProperty(
        name="Variable radius",
        description="Variable strut radius using 'radius' attribute from mesh",
        default=False,
    )

    charac_length: FloatProperty(
        name="Caracteristic length",
        description="Caracteristic length",
        default=10, 
        min=1.0e-2,
        unit='LENGTH',
    )


    icosub: IntProperty(
        name="Icosphere subdivisions",
        description="Number of subdivisions in the icosphere",
        default=2, 
        min=1,
    )

    move_dist: FloatProperty(
        name="Move distance",
        description="Maximum distance to move a lattice vertex to the border",
        default=1.0, 
        min=0.0,
        unit='LENGTH',
    )

    create_faces: BoolProperty(
        name="Create faces",
        description="Create an extra mesh with the border sleleton with faces",
        default=False,
    )

    min_angle: FloatProperty(
        name="Min angle",
        description="Minimal angle between two edges",
        default=math.radians(12.0),
        subtype='ANGLE',
        min=0.0,
        max=math.radians(90.0),
        )

    filter_angle: BoolProperty(
        name="Filter min angle",
        description="Remove edges that have a small angle to other edges",
        default=True,
    )

    add_dia_mod: BoolProperty(
        name="Add modifier",
        description="Add modifier to change strut diameter",
        default=True,
    )


classes = [
    SceneProperties,
    ui.VIEW3D_PT_lattices_generate,
    ui.VIEW3D_PT_lattice_Ico,
    ui.VIEW3D_PT_lattice_metaball,
    operators.ObjectIcoLattice,
    operators.ObjectMetaLattice,
    operators.ObjectCreateMeshLattice,
    operators.ObjectFillMeshLattice,
    operators.ObjectFilterLattice,
]

if operators.have_scipy:
    classes.append(operators.ObjectFillVoronoiLattice)

#def menu_func(self, context):
#    for cl in classes:
#        if cl is not SceneProperties:
#            self.layout.operator(cl.bl_idname)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lattice_mesh = PointerProperty(type=SceneProperties)
#    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.lattice_mesh
#    bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == "__main__":
    register()



