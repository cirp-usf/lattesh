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



import bpy, bmesh
from bpy.props import FloatProperty

from mathutils import Vector, Quaternion, Matrix
from mathutils.bvhtree import BVHTree
import numpy as np
import math
import os
from glob import glob
if bpy.app.version >= (4, 4, 0):
    import openvdb as vdb
else:
    import pyopenvdb as vdb

from itertools import combinations

try:
    from scipy.spatial import Voronoi, ConvexHull
    have_scipy = True
except ModuleNotFoundError:
    have_scipy = False


class ResolutionError(Exception):
    pass


class Vert:
    __slots__ = ('outside', 'onborder', 'index', 'close')
    def __init__(self, outside=None, onborder=None, index=None, close=None):
        self.outside = outside
        self.onborder = onborder
        self.index = index
        self.close = close
    def __str__(self):
        return 'V: out:{} on:{} ind:{} clo:{}'.format(self.outside, self.onborder, self.index, self.close)
    def __repr__(self):
        return self.__str__()

class Ico:
    def __init__(self, sub=2):
        # Create the icosphere
        self.bm = bmesh.new()
        bmesh.ops.create_icosphere(self.bm, subdivisions=sub, radius=1)

        verts = np.array([[x for x in v.co] for v in self.bm.verts], dtype=np.float32)
        tris = np.array([[v.index for v in f.verts] for f in self.bm.faces], dtype=np.int32)
        edges = np.array([[v.index for v in e.verts] for e in self.bm.edges], dtype=np.int32)

        mtris = verts[tris,:]
        mtris = mtris.sum(axis=1)
        mtris /= np.linalg.norm(mtris, axis=1, keepdims=True)
        edgelengths = np.linalg.norm(verts[edges[:,0],:] - verts[edges[:,1],:], axis=1)
        
        self.min_face_side = edgelengths.min()
        self.max_face_side = edgelengths.max()
        #Minimal angle between two verts
        self.min_angle = 2.0 * math.asin(self.min_face_side / 2.0)
        #print(sub, self.min_face_side, self.max_face_side, math.degrees(self.min_angle))

        self.afactor = self.min_angle / 10.0
        self.sub = sub
        self.verts = verts
        self.tris = tris
        self.mtris = mtris
        self.edges = edges
        self.nverts = len(verts)
        neigh = [set() for k in range(self.nverts)]
        for ed in self.edges:
            neigh[ed[0]].add(ed[1])
            neigh[ed[1]].add(ed[0])
        self.neighbors = [np.array(list(x), dtype=np.int32) for x in neigh]

    def get_bmesh(self):
        return self.bm.copy()


class Cell:
    def __init__(self, verts=None, edges=None):
        if verts is None:
            self.verts = np.zeros((0,3), dtype=np.float32)
            self.nverts = 0
        else:
            self.verts = verts
            self.nverts = verts.shape[0]
        if edges is None:
            self.edges = np.zeros((0,2), dtype=np.int32)
            self.nedges = 0
        else:
            self.edges = edges
            self.nedges = edges.shape[0]

        self.volumes = None
        self.edges_to_volumes = None

        self.check()

    def check(self):
        if self.nverts == 0:
            return
        if len(self.verts.shape) != 2 or self.verts.shape[1] != 3:
            raise ValueError("verts should be an N x 3 array")

        if len(self.edges.shape) != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges should be an M x 2 array")

        if self.verts.min() < 0.0 or self.verts.max() > 1.0:
            raise ValueError("All vertices' coordinates must be in [0.0, 1.0]")

        if self.edges.min() < 0 or self.edges.max() > self.nverts:
            raise ValueError("Edge(s) referencing inexistent vertex")

        if set(range(self.nverts)) != set(self.edges.flat):
            raise ValueError("Each vertex must be part of an edge")

    def from_file(self, fname):
        name = ''
        verts = []
        edges = []
        volumes = []
        edges_to_volumes = {}
        with open(fname) as f:
            for line in f:
                stripped = line.strip()
                if len(stripped) > 0 and stripped[0] != '#':
                    if len(name) == 0:
                        name = stripped
                    else:
                        sp = stripped.split(',')
                        if len(sp) == 3:
                            verts.append(list(map(float, sp)))
                        elif len(sp) == 2:
                            edges.append(list(map(int, sp)))
                        elif len(sp) > 3:
                            nvolumes = len(volumes)
                            volumes.append(list(map(float, sp[:3])))
                            vedges = list(map(int, sp[3:]))
                            for ed in vedges:
                                li = edges_to_volumes.setdefault(ed, [])
                                li.append(nvolumes)
                                
        self.verts = np.array(verts, dtype=np.float32)
        self.nverts = len(verts)
        self.edges = np.array(edges, dtype=np.int32)
        self.nedges = len(edges)
        self.check()
        self.name = name
        if len(volumes) > 0:
            self.volumes = np.array(volumes, dtype=np.float32)
            self.edges_to_volumes = edges_to_volumes



fpath = os.path.dirname(os.path.abspath(__file__)) 

nodes_file = os.path.join(fpath, 'nodes', 'nodes.blend')



Cells = {}
cell_items = []
cell_files = glob(os.path.join(fpath, 'cells/*.cell'))
cell_files.sort()
for k, cf in enumerate(cell_files):
    c = Cell()
    c.from_file(cf)
    Cells[c.name] = c
    cell_items.append((c.name, c.name, c.name, k))
    

def get_node_tree(ntree):
    ng = bpy.data.node_groups.get(ntree)
    if ng is None:
        print('Loading node tree {} from {}'.format(ntree, nodes_file))
        with bpy.data.libraries.load(nodes_file) as (data_from, data_to):
            data_to.node_groups = [ntree]
        ng = data_to.node_groups[0]
    return ng

def get_input_identifier(ntree, name):
    identifier = None
    if hasattr(ntree, 'interface'):
        identifier = ntree.interface.items_tree[name].identifier
    elif hasattr(ntree, 'inputs'):
        identifier = ntree.inputs[name].identifier

    return identifier

def make_npvoronoi(carac_length, dims, corner):
    """Makes a Voronoi lattice"""
    
    carac3 = carac_length ** 3
    padding = 2
    
    dims = dims + (2 * padding * carac_length)
    
    npoints = math.ceil(dims.prod() / carac3)
    
    npoints = max(npoints, 30) # create at least this amount of points
    
    print('voronoi npoints:', npoints)
    
    p = np.random.random((npoints,3))
    p = p * dims.reshape((1,3)) + (corner.reshape((1,3)) - padding * carac_length)


    vor = Voronoi(p)

    verts = vor.vertices

    # mark voronoi vertices outside extended bounding box
    lowerlims = corner.reshape((1,3)) - padding * carac_length
    upperlims = lowerlims + dims
    vertok = np.logical_and(np.all(verts > lowerlims, axis=1), np.all(verts < upperlims, axis=1))

    # new vertices
    new_verts = verts[vertok,:].copy()

    # build a table translating old indexes to new_verts
    okidx = np.zeros((len(verts),), dtype=np.int32)
    okidx[np.logical_not(vertok)] = -1
    newidx = okidx.cumsum() + np.arange(len(verts), dtype=np.int32)

    eds_s = set()

    for r in vor.ridge_vertices:
        if not -1 in r:
            for k in range(len(r) - 1):
                a = r[k]
                b = r[k + 1]
                if vertok[a] and vertok[b]:
                    na = newidx[a]
                    nb = newidx[b]
                    if na < nb:
                        eds_s.add((na,nb))
                    else:
                        eds_s.add((nb,na))

    edges = np.array([list(x) for x in eds_s], dtype=np.int32)
    return new_verts, edges



def make_nplattice(cell, cell_size, nelems, corner):
    """Constructs a lattice grid"""
    ncells = nelems.prod()
    verts = np.zeros((ncells * cell.nverts, 3), dtype=np.float32)
    edges = np.zeros((ncells * cell.nedges, 2), dtype=np.int32)
    cellcorner = np.zeros((1,3), dtype=np.float32)
    edgespos = np.ones((1,2), dtype=np.int32)

    # Generate a lattice with copied cells of unit size
    for i in range(nelems[0]):
        for j in range(nelems[1]):
            for k in range(nelems[2]):
                cellcorner[0, 0] = i
                cellcorner[0, 1] = j
                cellcorner[0, 2] = k
                baseidx = i + (j * nelems[0]) + (nelems[0] * nelems[1] * k)
                vertpos = baseidx * cell.nverts
                cellidx = baseidx * cell.nedges
                verts[vertpos : (vertpos + cell.nverts), :] = cellcorner + cell.verts
                edges[cellidx: (cellidx + cell.nedges), :] = cell.edges + edgespos * vertpos

    # Scale the whole lattice and translate it to corner
    verts = verts * cell_size + corner

    return verts, edges

def nplattice2meshdata(v, e):
    verts = [tuple(v[i,:]) for i in range(v.shape[0]) ]
    edges = [tuple(e[i,:]) for i in range(e.shape[0]) ]
    return verts, edges

def make_meshlattice(name, verts, edges, cellsize):
    mesh = bpy.data.meshes.new(name+'Mesh')
    mesh.from_pydata(verts, edges, [])
    
    bm = bmesh.new()
    
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1.0e-3*cellsize)
    return bm, mesh


def intersect(bm, obj, depsgraph):
    """Clips mesh lattice to object
    
    Returns a mesh where all vertices and edges are inside or on the surface
    of object. Edges that intercept the object are clipped.
    """
    tree = BVHTree.FromObject(obj, depsgraph)
    outside = set()
    inside = set()
    valencies = set()
    surfverts = set()
    cedges = []
    disttol = 5.0e-5
    double_edges = set()
    
    # Find the edges that cross object
    for e in bm.edges:
        p0 = e.verts[0].co
        p1 = e.verts[1].co
        d = p1 - p0
        dis = d.length
        loc, n, idx, hdis = tree.ray_cast(p0, d, dis)
        if hdis is not None:
            # check for multiple crossings
            loc2, n2, idx2, hdis2 = tree.ray_cast(p1, -d, dis)
            if loc2 and (loc - loc2).length > disttol / 2.0:
                double_edges.add(e)
            elif hdis > disttol and abs(dis - hdis) > disttol:
                r = hdis/dis
                cedges.append((e, r))

    print('edges', len(cedges))
    print('double edges', len(double_edges))

    # Delete double crossing edges
    if double_edges:
        bmesh.ops.delete(bm, geom=list(double_edges), context='EDGES')


    dper = dict()
    for k in cedges:
        dper[k[0]] = k[1]
    
    tmp = np.array(list(dper.values()))
    print('min dist', tmp.min(), 'max dist', tmp.max())

    # Split the found edges on crossing point
    bmesh.ops.bisect_edges(bm, edges=[ e[0] for e in cedges ], cuts=1, edge_percents=dper)['geom_split']

    # Find the vertices that are on the surface and identify their outside and inside edges based on 
    # surface normal
    for v in bm.verts:
        loc, n, idx, hdis = tree.find_nearest(v.co, 1.1*disttol)
        if loc is not None:
            surfverts.add(v)
            valencies.add(len(v.link_edges))
            for ed in v.link_edges:
                p1 = ed.other_vert(v)
                u_v = p1.co - v.co
                u_v = u_v / u_v.length * 0.2
                an1 = n.angle(p1.co - v.co)
                loc2, n2, idx, hdis = tree.find_nearest(p1.co, 1.1* ed.calc_length())
                if hdis and hdis > 0:
                    an2 = n2.angle((p1.co - loc2))
                else:
                    an2 = math.pi
                if  hdis and hdis > disttol and (an1 < math.pi / 2 + 1e-3 or an2 < math.pi / 2 + 1e-3) :
                    outside.add(ed)
                else:
                    inside.add(ed)

    print('Found', len(surfverts), 'surfverts')

    print('Valencies:', valencies)

    print('inside', len(inside), 'outside', len(outside))

    invalid_out = set([v for v in outside if not v.is_valid])
    outside = outside - invalid_out

    invalid_in = set([v for v in inside if not v.is_valid])
    inside = inside - invalid_in
    
    print('inv out', len(invalid_out), 'inv in', len(invalid_in), 'in out', len(inside.intersection(outside)))

    # Delete the outside edges from on-the-surface vertices
    bmesh.ops.delete(bm, geom=list(outside), context='EDGES')

    # Remove invalid surfverts
    invalid = set([v for v in surfverts if not v.is_valid])
    print('Found {} invalid surfverts'.format(len(invalid)))
    surfverts = surfverts - invalid
    
    # Find all vertices linked to the ones on the surface
    to_process = surfverts.copy()
    processed = set()
    while to_process:
        v = to_process.pop()
        for e in v.link_edges:
            other_v = e.other_vert(v)
            if other_v not in processed:
                to_process.add(other_v)
        processed.add(v)

    # Delete gemometry not  linked to surface vertices
    to_delete = set(bm.verts) - processed
    print(len(to_delete), 'to be deleted and', len(processed), 'internal vertices found')

    if len(to_delete) < len(bm.verts):
        # do not delete all vertices
        bmesh.ops.delete(bm, geom=list(to_delete), context='VERTS')


    # Leave only surface vertices selected
    bm.select_mode = {'VERT'}

    for v in bm.verts:
        v.select = False

    for v in surfverts:
        v.select = True

    bm.select_flush_mode()

    

def angle_dist(a, b):
    d = abs(b-a)
    return min(d, math.tau - d) # tau = 2 * pi


def ring_connect_faces(ang_a, ang_b):
    twopi =  math.tau
    na = len(ang_a)
    nb = len(ang_b)
    faces = np.zeros((na+nb, 3), dtype=np.int32)
    face_masq = np.zeros(faces.shape, dtype=bool)
    saidx = ang_a.argsort()
    sbidx = ang_b.argsort()
    sa = ang_a[saidx]
    sb = ang_b[sbidx]
    
    posa = 0
    nposa = 1
    dists = np.zeros((nb,2), dtype=ang_b.dtype)
    dists[:,0] = np.abs(sb - sa[0])
    dists[:,1] = twopi - dists[:,0]
    posb = dists.min(axis=1).argmin()
    nposb = (posb + 1) % nb
    ntris = 0
    ntrisa = 0
    ntrisb = 0
    
    while ntrisa < na and ntrisb < nb:
        dista = angle_dist(sa[posa], sb[nposb])
        distb = angle_dist(sb[posb], sa[nposa])
        if dista <= distb:
            faces[ntris,:] = [saidx[posa], sbidx[nposb], sbidx[posb]]
            face_masq[ntris,:] = [True, False, False]
            posb = (posb + 1) % nb
            nposb = (posb + 1) % nb
            ntrisb += 1
        else:
            faces[ntris,:] = [saidx[posa], saidx[nposa], sbidx[posb]]
            face_masq[ntris,:] = [True, True, False]
            posa = (posa + 1) % na
            nposa = (posa + 1) % na
            ntrisa += 1
        ntris += 1
    while ntrisa < na:
        faces[ntris,:] = [saidx[posa], saidx[nposa], sbidx[posb]]
        face_masq[ntris,:] = [True, True, False]
        posa = (posa + 1) % na
        nposa = (posa + 1) % na
        ntrisa += 1
        ntris += 1
    while ntrisb < nb:
        faces[ntris,:] = [saidx[posa], sbidx[nposb], sbidx[posb]]
        face_masq[ntris,:] = [True, False, False]
        posb = (posb + 1) % nb
        nposb = (posb + 1) % nb
        ntrisb += 1
        ntris += 1
        
    return faces, face_masq

vecz = np.array([0, 0, 1], dtype=np.float32)
eye = np.eye(3)
inv_eye = eye.copy()
inv_eye[1,1] = -1.0
inv_eye[2,2] = -1.0
twopi = math.pi * 2
min_crossnorm = 1e-3

def make_rotmatrix(x):
    """Make a rotation matrix that transform the given unity vector to the z axis"""
    c_angle = x @ vecz
    raxis = np.cross(x, vecz)
    if np.linalg.norm(raxis) < min_crossnorm:
        if c_angle > 0.0:
            rotmat = eye
        elif c_angle < 0.0:
            rotmat = inv_eye
    else:
        rotmat = np.array(Matrix.Rotation(math.acos(c_angle), 3, raxis))
    return rotmat


def make_rotmatrix2(z, xp):
    x = xp - (xp @ z)*z
    y = np.cross(z, x)
    m = np.concatenate((x.reshape(1,3), y.reshape(1,3), z.reshape(1,3)), axis=0)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def ico_compute_node(ev, ico):
    # minimum distance between generated vertices
    # angle factor
    min_per = 2e-2
    theta_crit = math.atan(0.5 / 3)
    bm = ico.get_bmesh()
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    verts = ico.verts.copy()
    p = verts @ ev.T
    np.clip(p, -1.0, 1.0, out=p)
    psort = p.argsort(axis=1)[:, ::-1].copy()
    linidx = np.arange(ico.nverts, dtype=int)
    pbest = p[linidx, psort[:,0]].copy()
    close_ev = pbest > 0.0
    to_keep = np.logical_not(close_ev)
    nverts = ico.nverts

    #print('Non edge verts:', np.count_nonzero(to_keep), 'Max angle:', math.degrees(math.acos(pbest.min())))
    recompute = True
    while recompute:
        old_nverts = nverts
        to_split = []
        new_keep = []
        percentages = {}
        recompute = False
        for ed in bm.edges:
            ed_verts = [v.index for v in ed.verts]
            if not np.all(close_ev[ed_verts]):
                continue
            ev_idx = psort[ed_verts, 0]
            if ev_idx[0] == ev_idx[1]:
                continue
            projs0 = p[ed_verts[0],ev_idx[0]] - p[ed_verts[0],ev_idx[1]]
            projs1 = p[ed_verts[1],ev_idx[0]] - p[ed_verts[1],ev_idx[1]]
            per = projs0 / (projs0 - projs1)
            if per < 0 or per > 1.0:
                print('=======')
                print('Failure in convex combination')
                print(per)
                print(p[ed_verts, :])
                print(psort[ed_verts, :])
                continue
            if per < min_per:
                to_keep[ed_verts[0]] = True
                continue
            if per > 1 - min_per:
                to_keep[ed_verts[1]] = True
                continue
            to_split.append(ed)
            percentages[ed] = per
            new_vert = per * verts[ed_verts[0]] + (1.0 - per) * verts[ed_verts[1]]
            new_p = ev @ new_vert
            new_max = new_p.argmax()
            if new_max != ev_idx[0] and new_max != ev_idx[1]:
                recompute = True
                new_keep.append(False)
            else:
                new_keep.append(True)
        #print('Nedges to split:', len(to_split))
        bmesh.ops.subdivide_edges(bm, edges=to_split, cuts=1, edge_percents=percentages)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        nnewverts = len(to_split)
        nverts = nnewverts + old_nverts
        verts.resize((nverts, 3))
        to_keep.resize(nverts)
        close_ev.resize(nverts)
        to_keep[old_nverts:] = new_keep
        close_ev[old_nverts:] = np.logical_not(to_keep[old_nverts:])
        p.resize((nverts, len(ev)))
        psort.resize((nverts, len(ev)))
        #print(nverts, ico.nverts, nnewverts, len(bm.verts))
        for k in range(old_nverts, nverts):
            bm.verts[k].co.normalize()
            verts[k, :] = bm.verts[k].co
        p[old_nverts:nverts, :] = verts[old_nverts:nverts, :] @ ev.T
        np.clip(p, -1.0, 1.0, out=p)
        psort[old_nverts:nverts, :] = p[old_nverts:nverts, :].argsort(axis=1)[:,::-1]


    linidx = np.arange(nverts, dtype=int)
    pbest = np.clip( np.arccos(p[linidx, psort[:,0]]), ico.min_angle / 3, math.pi / 2)

    factor = 1.0 / np.sin(pbest)
    verts *= factor[:,np.newaxis]
    tris = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int32)
    ed_verts = []
    ed_ind_mask = np.array([[1,2], [2, 0], [0, 1]])
    for node_edge in range(len(ev)):
        calc_edges = {}
        recalc = True
        count = 0
        while recalc:
            count += 1
            cand = np.logical_and(psort[:,0] == node_edge, np.logical_not(to_keep))
            if not np.count_nonzero(cand) > 0:
                raise ResolutionError
            border_tris = np.logical_and(np.count_nonzero(cand[tris], axis=1) == 1, np.count_nonzero(to_keep[tris], axis=1) == 2)
            recalc = False
            for tri in tris[border_tris,:]:
                vpos = np.nonzero(cand[tri])[0]
                tri_vert = tri[vpos]
                tri_ed = tri[ed_ind_mask[vpos,:][0]]
                t_ed = tuple(tri_ed)
                if t_ed not in calc_edges:
                    lz = verts[tri_ed, :].sum(axis=0)
                    lz = lz / np.linalg.norm(lz)
                    a = (verts[tri_ed[0],:] - verts[tri_ed[1]]) / 2
                    rotmat = make_rotmatrix2(lz, ev[node_edge,:])[:2, :]
                    ar = rotmat @ a
                    phi = math.atan2(ar[1], ar[0])
                    calc_edges[t_ed] = phi
                phi = calc_edges[t_ed]
                if phi < theta_crit or phi > math.pi - theta_crit:
                    to_keep[tri_vert] = True
                    recalc = True
            
        border_verts = tris[border_tris,:].flatten()
        ed_verts.append(set(border_verts[to_keep[border_verts]]))

    bm.free()
    tris_to_keep = np.all(to_keep[tris], axis=1)
    verts_trans = to_keep.astype(int).cumsum() - 1
    rverts = verts[to_keep,:].copy()
    rtris = verts_trans[tris[tris_to_keep,:]].copy()
    red_verts = [verts_trans[np.array(list(x))].copy() for x in ed_verts]

    return rverts, rtris, red_verts


def createMeshIco(target, depsgraph, r=0.5, var_radius=False, icosub=2, add_index=False, add_mod=False):
    """Gives a volume to each edge in target using cylinders of radius r"""
    quantization = 100
    tmesh = target.evaluated_get(depsgraph).data
    name = target.name + " Ico"
    nverts = len(tmesh.vertices)
    nedges = len(tmesh.edges)
    index_attrib_name = 'skeleton_index'
    
    verts = np.empty(nverts * 3, dtype=np.float32)
    edges = np.empty(nedges * 2, dtype=np.int32)
    var_radius = var_radius and 'radius' in tmesh.attributes
    if var_radius :
        radius = np.empty(nverts, dtype=np.float32)
        tmesh.attributes['radius'].data.foreach_get('value', radius)
        dia = ' {:.1f}-{:.1f}'.format(radius.min() * 2, radius.max() * 2)
        # prescale the radius
    else:
        radius = np.full(nverts, r, dtype=np.float32)
        dia = ' {:.1f}'.format(r * 2)
    name += dia

    tmesh.vertices.foreach_get('co', verts)
    tmesh.edges.foreach_get('vertices', edges)
    
    verts.shape = (nverts,3)
    edges.shape = (nedges,2)
    
    vec_edges = verts[edges[:,1],:] - verts[edges[:,0],:]
    edge_len = np.linalg.norm(vec_edges, axis=1, keepdims=True)
    vec_edges /= edge_len
    
    icospheres = [Ico(x + icosub) for x in range(4)]
    
    
    # Quantize edge orientations
    qedges = (vec_edges * quantization).astype(np.int32)
    
    dqedges = {}
    for k in range(nedges):
        qv = tuple(qedges[k,:])
        dqedges.setdefault(qv,[]).append(k)

    edge_t = np.zeros(nedges, dtype=np.int32)
    edge_types = []
    for k, t in enumerate(dqedges.keys()):
        edge_types.append(t)
        edge_t[dqedges[t]] = k


    mesh = bpy.data.meshes.new(name+'Mesh')

    print(f'{nedges} total edges, {len(dqedges)} quantized orientations. ')

    
    # Generate all possible edge orientations, including reverse
    tempset = set()
    for k in dqedges.keys():
        tempset.add(k)
        bw = -np.array(list(k))
        tempset.add(tuple(bw))

    dvecs = {}
    listdir = list(tempset)
    for k, v in enumerate(listdir):
        dvecs[v] = k

    # Record all edges and their directions w.r.t. each node (vertex)
    # nodes is a list containing, for every vertex, a list of pairs [ind, dir],
    # where ind is the index of an edge that is connected with the vertex and
    # dir is true if the edge starts in this vertex and false otherwise
    nodes = [None] * nverts
    for k in range(nedges):
        v0, v1 = edges[k,0], edges[k,1]
        t = (k, True)
        if nodes[v0] is None:
            nodes[v0] = [t]
        else:
            nodes[v0].append(t)
        t = (k, False)
        if nodes[v1] is None:
            nodes[v1] = [t]
        else:
            nodes[v1].append(t)

    # Classify all nodes based on their (discretized) edge orientations
    node_types = {}
    single_nodes = set()
    for k in range(nverts):
        if nodes[k] is None:
            # Ignore isolated vertices
            continue
        if len(nodes[k]) == 1:
            single_nodes.add(k)
        cand_edges = []
        cand_dir = []
        for kk, eddir in nodes[k]:
            cand_edges.append(kk)
            cand_dir.append(eddir)
        if len(cand_edges) == 0:
            continue
        node_edges = np.array(cand_edges, dtype=np.int32) # list of edge indeces
        edge_dir = np.array(cand_dir, dtype=np.int32)
        edge_vec = qedges[node_edges, :].copy()
        edge_vec[np.flatnonzero(np.logical_not(edge_dir)),:] *= -1 # flip the edge vector so that all point away from vertex
        dir_keya = np.array([dvecs[tuple(x)] for x in edge_vec]) # retrieve the index of each quantitized vector in listdir
        sd = dir_keya.argsort()
        dir_key = tuple(dir_keya[sd])
        node_types.setdefault(dir_key, []).append(k)
        temp = np.stack((node_edges, edge_dir.astype(np.int32)), axis=1)
        nodes[k] = temp[sd,:] # sort the node's list of edges according to edge position in dict's keys

    print(nverts, 'total vertices,', len(node_types), 'distinct nodes,', len(single_nodes), 'single nodes.')



    total_verts = 0
    total_faces = 0
    processed_nodes = []
    nr_nodetypes = len(node_types)
    for k, (nt, nodelist) in enumerate(node_types.items()):
        if nr_nodetypes > 100:
            if k % 10 == 0:
                print(f'Processing node {k} of {nr_nodetypes}', end='\r')
        # Reconstruct edge vectors based on quantitized vectors
        ev = np.array([listdir[d] for d in nt], dtype=np.float32) / quantization

        # Compute node geomery using an icosphere
        node_success = False
        for sphere in icospheres:
            try:
                node_verts, node_faces, node_edverts= ico_compute_node(ev, sphere)
            except ResolutionError:
                pass
            else:
                node_success = True
                if sphere.sub != icosub:
                    print('\nNew node subdivision:', sphere.sub)
                break
        if not node_success:
            raise ResolutionError('Computing node failed')
        processed_nodes.append((node_verts, node_faces, node_edverts, nt, nodelist, ))
        total_verts += len(node_verts) * len(nodelist)
        total_faces += len(node_faces) * len(nodelist)

    newverts = np.empty((total_verts, 3), dtype=np.float32)
    newfaces = np.empty((total_faces, 3), dtype=np.int32)
    if add_index:
        newindex = np.empty(total_verts, dtype=np.int32)

    vert_count = 0
    face_count = 0
    edgefaces = np.zeros((nedges, 2), dtype=np.int32)
    ed_nd = np.zeros((nedges, 2, 2), dtype=np.int32)
    for k, (node_verts, node_faces, node_edverts, nt, nodelist) in enumerate(processed_nodes):
        valency = len(node_edverts)
        verts_per_node = len(node_verts)
        faces_per_node = len(node_faces)
        ed_pos = np.arange(valency, dtype=np.int32)
        if valency:
            for n in nodelist:
                # Translate the vertices of each node type to the actual position of the nodes and
                # insert them in newverts
                if verts_per_node:
                    temp_verts = node_verts * radius[n]
                    newverts[vert_count:vert_count + verts_per_node, :] = temp_verts + verts[n,:]
                    if add_index:
                        newindex[vert_count:vert_count + verts_per_node] = n
                # Insert the node type faces in newfaces for each of the node instances
                if faces_per_node:
                    newfaces[face_count:face_count + faces_per_node, :] = node_faces + vert_count
                edgefaces[nodes[n][:,0], 1 - nodes[n][:,1]] = vert_count
                ed_nd[nodes[n][:,0],0, 1 - nodes[n][:,1]] = k
                ed_nd[nodes[n][:,0],1, 1 - nodes[n][:,1]] = ed_pos
                vert_count += verts_per_node
                face_count += faces_per_node
        else:
            print('Edgeless node found. Node list:', nodelist)

    ed_nd_dict = {}
    for k in range(nedges):
        node_a = tuple(ed_nd[k, :, 0])
        node_b = tuple(ed_nd[k, :, 1])
        edkey = tuple(sorted([node_a, node_b]))
        edir = int(node_a == edkey[0])
        ed_nd_dict.setdefault(edkey, []).append([k, edir])

    print('\nEdge node types:', len(ed_nd_dict))
    
    ed_nd_faces = {}
    ed_nd_nfaces = 0
    for ed_ty in ed_nd_dict.keys():
        (na, fa), (nb, fb) = ed_ty
        node_verts, node_faces, node_edverts, nt, nodelist = processed_nodes[na]
        ed_dir = np.array(listdir[nt[fa]], dtype=np.float32) / quantization
        ed_dir /= np.linalg.norm(ed_dir)
        nedidxa = node_edverts[fa]
        nva = node_verts[nedidxa, :]
        node_verts, node_faces, node_edverts, nt, nodelist = processed_nodes[nb]
        nedidxb = node_edverts[fb]
        nvb = node_verts[nedidxb, :]
        rotmat = make_rotmatrix(ed_dir)
        rotmat = rotmat.T[:,:2].copy()
        proj_verts = nva @ rotmat
        ang_a = np.arctan2(proj_verts[:,1], proj_verts[:,0])
        proj_verts = nvb @ rotmat
        ang_b = np.arctan2(proj_verts[:,1], proj_verts[:,0])
        ed_fa, face_masq = ring_connect_faces(ang_a, ang_b)
        other_masq = np.logical_not(face_masq)
        ed_fanew = np.zeros(ed_fa.shape, dtype=np.int32)
        try:
            ed_fanew[face_masq] = nedidxa[ed_fa[face_masq]]
            ed_fanew[other_masq] = nedidxb[ed_fa[other_masq]]
            ed_nd_faces[ed_ty] = [ed_fanew, face_masq]
            ed_nd_nfaces += len(ed_nd_dict[ed_ty]) * len(ed_fa)
        except Exception:
            print('ed_ty', ed_ty)
            print('nedidxa')
            print(nedidxa)
            print(face_masq)
            print('ed_fa')
            print(ed_fa[face_masq])
            print('nva.shape', nva.shape)
            raise

    newfaces.resize((total_faces + ed_nd_nfaces, 3))
    print('Nb. of edge node faces:', ed_nd_nfaces)
        

    processed_faces = 0
    for ed_ty in ed_nd_dict.keys():
        edfaces, face_masq = ed_nd_faces[ed_ty]
        other_facemasq = np.logical_not(face_masq)
        ed_nfaces = len(edfaces)
        faceoffsets = np.zeros(edfaces.shape, dtype=np.int32)
        for edidx, edir in ed_nd_dict[ed_ty]:
            faceoffsets[face_masq] = edgefaces[edidx, 1-edir]
            faceoffsets[other_facemasq] = edgefaces[edidx, edir]
            newfaces[total_faces + processed_faces : total_faces + processed_faces + ed_nfaces,:] = edfaces + faceoffsets
            processed_faces += ed_nfaces

    mesh.from_pydata(newverts, [], list(newfaces))
    if add_index:
        index_attrib = mesh.attributes.new(index_attrib_name, 'INT', 'POINT')
        index_attrib.data.foreach_set('value', newindex)
    newobj  = bpy.data.objects.new(name, mesh)

    # Place the lattice at the same position and attitude of original object       
    newobj.matrix_world = target.matrix_world.copy()

    if add_index and add_mod:
        # Add Change diameter modifier
        ng = get_node_tree('Change diameter')
        identifier = get_input_identifier(ng, 'Target')
        if identifier is not None:
            mod = newobj.modifiers.new('Change diameter', 'NODES')
            mod.node_group = ng
            mod[identifier] = target

    # Link object to scene
    bpy.context.collection.objects.link(newobj)


def createMetaball(target,  r=0.5, p=0.9):
    if target.type != 'MESH':
        return
    
    name = target.name + "Rod {:.1f}".format(r * 2)
    print("Creating", name)

    metaball = bpy.data.metaballs.new(name+'Meta')
    obj = bpy.data.objects.new(name, metaball)
    bpy.context.collection.objects.link(obj)
    stiffness = 10.0
    
    # clip p
    p = min(p, 0.99);
    p = max(p, 0.01)

    # Compute the threashold so that mesh radius is 
    # exacltly r
    metaball.threshold = (1 - p ** 2) ** 3 * stiffness
    
    radius = r / p
    
    vec_x = Vector((1.0, 0.0, 0.0))

    for v in target.data.vertices:
        location = v.co.copy()

        element = metaball.elements.new()
        element.co = location
        element.radius = radius
        element.stiffness = stiffness

    for e in target.data.edges:
        v0 = target.data.vertices[e.vertices[0]].co
        v1 = target.data.vertices[e.vertices[1]].co
        edge = v1 - v0
        edge_len = edge.length

        if edge_len > r:
            edge_middle = (v1 + v0) / 2.0
            element = metaball.elements.new()
            element.co = edge_middle
            element.radius = radius
            element.stiffness = stiffness

            if edge_len > 2 * r:
                element.type = 'CAPSULE'
                element.size_x = (edge_len - 2 * r) / 2.0
                angle = edge.angle(vec_x)
                if abs(angle) > 1e-3:
                    axis = edge.cross(vec_x).normalized()
                    rot = Quaternion(axis, -angle).normalized()
                    element.rotation = (rot.w, rot.x, rot.y, rot.z)
                    
                
    # Place the lattice at the same position and attitude of original object       
    obj.matrix_world = target.matrix_world.copy()

    return obj


def orient_simplex(simplex, verts, normal):
    v1 = verts[simplex[1], :] - verts[simplex[0], :]
    v2 = verts[simplex[2], :] - verts[simplex[1], :]
    if np.cross(v1, v2) @ normal < 0:
        newsimplex = simplex[[0, 2, 1]]
    else:
        newsimplex = simplex.copy()
    return newsimplex

def compute_levelset(mesh, scale, move_dist=None):
    if move_dist is None:
        hwidth = 3
    else:
        hwidth = max(3.0, int(move_dist / scale) + 2.0)
    nmverts = len(mesh.vertices)
    nmtris = len(mesh.loop_triangles)
    mverts = np.zeros((nmverts * 3), dtype = np.float32)
    mtris = np.zeros((nmtris * 3), dtype = np.int32)
    mesh.vertices.foreach_get('co', mverts)
    mesh.loop_triangles.foreach_get('vertices', mtris)
    mverts.shape = (nmverts, 3)
    mtris.shape = (nmtris, 3)
    tr = vdb.createLinearTransform(scale)
    levelset = vdb.FloatGrid.createLevelSetFromPolygons(mverts, triangles=mtris, transform=tr, halfWidth=hwidth)
    return levelset

def is_outside(pos, levelset, move_dist):
    gridsize = levelset.transform.voxelSize()[0]
    pos_i = np.array(levelset.transform.worldToIndex(pos.astype(np.float64)))
    pos_ijk = np.floor(pos_i).astype(np.int32)
    neighbors = np.zeros((2,2,2), dtype=np.float32)
    levelset.copyToArray(neighbors, ijk = pos_ijk)
    # TODO Add trinlinear interpolation?
    nmin, nmax = neighbors.min(), neighbors.max()
    outside = nmin > move_dist
    dist = min(abs(nmin), abs(nmax))
    close_to_border = dist <= move_dist
    onborder = not outside and dist < gridsize 
    return  outside, onborder, close_to_border

def find_closest(pos, tree, move_dist):
    npos, normal, ind, dist = tree.find_nearest(pos, move_dist * 10.0)
    if npos is not None:
        npos = np.array(npos, dtype=np.float32)
    return npos

def make_faces(verts, edges, border_verts, faces):
    ind = border_verts > 0
    translate = np.cumsum(ind) - 1
    newverts = verts[ind]
    newedges = translate[edges[np.all(border_verts[edges], axis=1), :]]
    newfaces = [translate[f] for f in faces]
    return newverts, newedges, newfaces

def fill_mesh(mesh, pos, pos_label, size, lattice_cell, connect_boundary=True, move_dist=1, create_faces=False):
    # Compute a high resolution level set
    grid_size = max(0.2, move_dist / 5)
    grid_size = min(grid_size, size / 10)
    move_dist = max(move_dist, grid_size)
    has_regions = lattice_cell.volumes is not None
    faces = []
    print('Computing high density levelset')
    ls_high = compute_levelset(mesh, grid_size, move_dist)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    tree = BVHTree.FromBMesh(bm)
    verts = []
    edges = []
    visited_verts = {}
    added_edges = {}
    border_verts = set()
    border_volumes = {}
    computed_intersections = {}
    dsize = 10.0 / size # discretize verts coordinates in 10x10x10 grid inside cell
    cell_scaled = lattice_cell.verts * size
    ncells = len(pos_label)
    ninterior = np.count_nonzero(pos_label == 1)
    nborder = 0
    print('Processing cells')
    for cand_ind in range(ncells):
        if ncells > 500:
            if cand_ind % 50 == 0:
                print('Processing cell {} of {}'.format(cand_ind, ncells), end='\r')
        p = pos[cand_ind, :]
        is_border = pos_label[cand_ind] == 2
        this_cell_present = False
        cell_verts = cell_scaled + np.reshape( p, (1, 3))
        if is_border and has_regions:
            local_volumes = [set() for x in lattice_cell.volumes]
            volume_pos = lattice_cell.volumes * size
            vpos = volume_pos + np.reshape( p, (1, 3))
            vposd = [tuple(np.round(x * dsize).astype(int)) for x in vpos]
        else:
            local_volumes = []
            vpos = []
            vposd = []

        local_border_volumes = set()
        vs = np.zeros((2, 3), dtype=np.float32)
        for ed_ind, ed in enumerate(lattice_cell.edges):
            vs[:,:] = cell_verts[ed, :]
            vds = (tuple(np.round(vs[0] * dsize).astype(int)), tuple(np.round(vs[1] * dsize).astype(int)))
            vertos = []
            for v, vd in zip(vs, vds):
                if vd in visited_verts:
                    verto = visited_verts[vd]
                else:
                    verto = Vert()
                    outside, onborder, close_to_border = is_outside(v, ls_high, move_dist)
                    verto.outside = outside
                    verto.onborder = onborder
                    verto.close = close_to_border
                    visited_verts[vd] = verto
                vertos.append(verto)

            if vertos[0].outside and vertos[1].outside:
                # edge is completely outside, ignore it
                continue
            if not vertos[0].outside and not vertos[1].outside:
                # entire edge is inside or on the border, add to the list, if not already there
                this_cell_present = True
                for verto, v in zip (vertos, vs):
                    if verto.index is None:
                        if verto.close or verto.onborder:
                            vnew = find_closest(v, tree, move_dist)
                            if vnew is not None:
                                v[:] = vnew
                                verto.onborder = True
                                verto.outside = False
                                verto.close = True
                            else:
                                print('Move to border failed to find a new point')
                        nverts = len(verts)
                        verts.append(v.copy())
                        verto.index = nverts
                        if verto.onborder:
                            border_verts.add(nverts)
                # Add this vertices' indices to their volumes
                if is_border and has_regions:
                    for k in lattice_cell.edges_to_volumes[ed_ind]:
                        local_volumes[k].add(vertos[0].index)
                        local_volumes[k].add(vertos[1].index)
                        if vertos[0].onborder or vertos[1].onborder:
                            local_border_volumes.add(k)
                edge_tuple = tuple(sorted([vertos[0].index, vertos[1].index]))
                if edge_tuple not in added_edges:
                    nedges = len(edges)
                    edges.append(np.array(edge_tuple, dtype=np.int32))
                    added_edges[edge_tuple] = nedges
            else:
                if vertos[0].outside:
                    vin, vind, vout, voutd = vs[1], vds[1], vs[0], vds[0]
                    inside_onborder = vertos[1].close or vertos[1].onborder
                else:
                    vin, vind, vout, voutd = vs[0], vds[0], vs[1], vds[1]
                    inside_onborder = vertos[0].close or vertos[0].onborder

                if inside_onborder:
                    # vin is effectively on surface, ignore this edge
                    continue
                intersect = (*vind, *voutd)
                if intersect not in  computed_intersections:
                    computed_intersections[intersect] = None
                    direction = vout - vin
                    edge_length = np.linalg.norm(direction)
                    hit_pos, normal, ind, dist = tree.ray_cast(vin, direction / edge_length, edge_length)
                    if hit_pos is None:
                        hit_pos, normal, ind, dist = tree.ray_cast(vin, direction / edge_length, edge_length * 2)
                        if dist is not None:
                            hdist = dist / edge_length
                        else:
                            hdist = 'inf'
                        print('Warning: error in ray cast. Ignoring edge. Distance: {}'.format(hdist), vertos)
                        print(vin, np.linalg.norm(vin), vout, np.linalg.norm(vout))
                        continue
                    vertin = visited_verts[vind]
                    if vertin.index is None:
                        nverts = len(verts)
                        verts.append(vin.copy())
                        vertin.index = nverts

                    nverts = len(verts)
                    verts.append(np.array(hit_pos, dtype=np.float32))
                    border_verts.add(nverts)
                    edlen = np.linalg.norm(verts[nverts]- vin)
                    if  edlen< move_dist:
                        print('Short edge:',edlen, vin)
                        print(is_outside(vin, ls_high, move_dist))
                        print(vertos)
                        

                    edge_tuple = (vertin.index, nverts)
                    nedges = len(edges)
                    edges.append(np.array(edge_tuple, dtype=np.int32))
                    added_edges[edge_tuple] = nedges
                    computed_intersections[intersect] = edge_tuple

                inter_edge = computed_intersections[intersect]
                if inter_edge is not None:
                    this_cell_present = True
                if has_regions and inter_edge is not None:
                    for k in lattice_cell.edges_to_volumes[ed_ind]:
                        local_volumes[k].add(inter_edge[0])
                        local_volumes[k].add(inter_edge[1])
                        local_border_volumes.add(k)
        if is_border and this_cell_present:
            # count the number of border cells that were actually included
            nborder += 1


        # All edges processed. Compute border volumes
        for k in local_border_volumes:
            vind = vposd[k]
            vlist = border_volumes.setdefault(vind, set())
            vlist.update(local_volumes[k])

    verts = np.array(verts, dtype=np.float32)

    # Process border vertices
    if connect_boundary and has_regions:
        border_to_process = len(border_volumes)
        print('\nProcessing border regions')
        hull_edges = {}
        border_simplices = {}
        simplices_to_connect = {}
        edges_to_remove = {}
        for region, volume_verts in enumerate(border_volumes.values()):
            if border_to_process > 500:
                if region % 100 == 0:
                    print('Processing region {} of {}'.format(region, border_to_process), end='\r')
            vol_verts_on_border = volume_verts.intersection(border_verts)
            n_onborder = len(vol_verts_on_border)
                
            if n_onborder < 2:
                continue
            elif n_onborder < 4:
                # There are only 2 or 3 border vertices in this volume, connect them
                # If 3, the tri won't get a face, leave that for fill holes
                v = sorted(vol_verts_on_border)
                for ed in combinations(v, 2):
                    if ed not in added_edges:
                        nedges = len(edges)
                        edges.append(np.array(ed, dtype=np.int32))
                        added_edges[ed] = nedges
            else:
                vinds = np.array(list(volume_verts))
                vpoints = verts[vinds, :]
                is_border = np.array([x in border_verts for x in vinds])
                
                try:
                    hull = ConvexHull(vpoints)
                except Exception:
                    print('Error computing convex hull', len(vinds))
                    print('volume_verts', volume_verts)
                    print(vpoints)
                    continue
                # find the simplices with only border vertices
                is_sim_onborder = np.all(is_border[hull.simplices], axis=1)
                sim_on_border = np.nonzero(is_sim_onborder)
                hull_edges.clear()
                border_simplices.clear()
                edges_to_remove.clear()
                simplices_to_connect.clear()
                
                # find all border edges
                for k, simplex in enumerate(hull.simplices):
                    facet = np.sort(simplex[is_border[simplex]])
                    if len(facet) >= 2:
                        for ed in combinations(facet, 2):
                            indlist = hull_edges.setdefault(ed, [])
                            indlist.append(k)
                for ed, facet_numbers in hull_edges.items():
                    if len(np.intersect1d(np.array(facet_numbers), sim_on_border)) > 1:
                        # edge is between two triangles on the surface
                        # compute the angle between these tris and ignore the edge 
                        # if the angle is too close to 180
                        if len(facet_numbers) != 2:
                            print('Wrong number of facets for edge')
                            print('Edge:', ed, 'Facet numbers:', facet_numbers)
                            print('Simplices:')
                            print(hull.simplices)
                        else:
                            #tris = vpoints[hull.simplices [facet_numbers]]
                            # Normals
                            nm = hull.equations[facet_numbers, :3]
                            proj = nm[0, :] @ nm[1, :]
                            
                            if abs(proj) > math.cos(math.radians(40)):
                                # Faces are less than 40 degrees apart, ignore edge
                                edges_to_remove[ed] = facet_numbers
                                for fa in facet_numbers:
                                    simplices_to_connect.setdefault(fa, set()).add(ed)
                                continue
                        
                    edge_tuple = tuple(sorted(vinds[list(ed)]))
                    if edge_tuple not in added_edges:
                        nedges = len(edges)
                        edges.append(np.array(edge_tuple, dtype=np.int32))
                        added_edges[edge_tuple] = nedges

                # Add faces
                if create_faces:
                    for k, simplex in enumerate(hull.simplices):
                        if is_sim_onborder[k]:
                            border_simplices[k] = orient_simplex(simplex, vpoints, hull.equations[k, :3] )
                            if k not in simplices_to_connect:
                                faces.append(vinds[border_simplices[k]])
                    regions = []
                    to_connect = set(simplices_to_connect.keys())
                    while len(to_connect) > 0:
                        first_tri = to_connect.pop()
                        face_set = set()
                        face_set.add(first_tri)
                        to_search = set()
                        for ed in simplices_to_connect[first_tri]:
                            for tri in edges_to_remove[ed]:
                                if tri not in face_set:
                                    to_search.add(tri)
                        while len(to_search) > 0:
                            this_tri = to_search.pop()
                            to_connect.discard(this_tri)
                            face_set.add(this_tri)
                            for ed in simplices_to_connect[this_tri]:
                                for tri in edges_to_remove[ed]:
                                    if tri not in face_set:
                                        to_search.add(tri)
                        regions.append(face_set)
                    if len(edges_to_remove) > 0:
                        forbidden_edges = set(edges_to_remove.keys())
                        forbidden_edges.update([(x[1], x[0]) for x in forbidden_edges])
                    for face_set in regions:
                        edge_pool = []
                        full_pool = []
                        for k in face_set:
                            for kk in range(3):
                                ed = (border_simplices[k][kk], border_simplices[k][(kk + 1) % 3])
                                full_pool.append(ed)
                                if ed not in forbidden_edges:
                                    edge_pool.append(ed)
                        ep_orig = edge_pool.copy()
                        if len(edge_pool) == 0:
                            continue
                        ed = edge_pool.pop()
                        new_face = [ed[0]]
                        while len(edge_pool) > 1:
                            new_face.append(ed[1])
                            for ind, ned in enumerate(edge_pool):
                                if ned[0] == ed[1]:
                                    break
                            ed = edge_pool.pop(ind)
                        new_face.append(ed[1])
                        if edge_pool[0][1] != new_face[0]:
                            print('error:', new_face, edge_pool, ep_orig)
                        faces.append(vinds[new_face])

    # end if has_regions
    total = nborder + ninterior
    print('\nMesh has {} cells. Interior: {} ({:.1f}%) Border: {} ({:.1f}%)'.format(total, ninterior, 
          (ninterior / total) * 100 , nborder, (nborder / total) * 100))
    print('Original borders:', np.count_nonzero(pos_label==2))
    edges = np.array(edges, dtype=np.int32)
    edge_lengths = np.linalg.norm(verts[edges[:,0]] - verts[edges[:,1]], axis=1)
    min_len = edge_lengths.min()
    max_len = edge_lengths.max()
    median_len = np.median(edge_lengths)
    
    report_data = {'ninterior':ninterior, 'nborder':nborder, 'min_len':min_len, 'max_len':max_len, 'median_len':median_len}
    return verts, edges, border_verts, faces, report_data


def clean_edges(verts, edges, angle):
    nverts = len(verts)
    nedges = len(edges)
    cos_angle = math.cos(angle)
    vec_edges = verts[edges[:,1],:] - verts[edges[:,0],:]
    edge_len = np.linalg.norm(vec_edges, axis=1)
    vec_edges /= edge_len[:, np.newaxis]

    v2e = [[] for x in range(nverts)]
    for k, ed in enumerate(edges):
        v2e[ed[0]].append(k)
        v2e[ed[1]].append( -k)

    edmask = np.ones(nedges, dtype=bool)

    for k, elist in enumerate(v2e):
        lindex = np.abs(elist)
        lsign = np.sign(elist)
        lvecs = vec_edges[lindex] * lsign[:, np.newaxis]
        for pair in combinations(range(len(elist)), 2):
            sprod = lvecs[pair[0],:] @ lvecs[pair[1],:]
            if sprod > cos_angle:
                if edge_len[lindex[pair[0]]] > edge_len[lindex[pair[1]]]:
                    toremove = lindex[pair[0]]
                else:
                    toremove = lindex[pair[1]]
                edmask[toremove] = False
                #print('Edge to remove', toremove)

    return  edges[edmask, :]


class ObjectMetaLattice(bpy.types.Operator):
    """Metaball Lattice"""
    bl_idname = "object.metalattice"
    bl_label = "Meta Lattice"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'


    def execute(self, context):
        lattice_mesh = context.scene.lattice_mesh
        createMetaball(context.active_object, lattice_mesh.radius, lattice_mesh.r_ratio)
        return {'FINISHED'}


class ObjectIcoLattice(bpy.types.Operator):
    """Lattice Volume (Icosphere)"""
    bl_idname = "object.icolattice"
    bl_label = "Lattice Volume (Ico)"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'


    def execute(self, context):
        lattice_mesh = context.scene.lattice_mesh
        depsgraph = context.evaluated_depsgraph_get()
        createMeshIco(context.active_object,
                      depsgraph, 
                      lattice_mesh.radius, 
                      lattice_mesh.variable_radius, 
                      lattice_mesh.icosub, 
                      lattice_mesh.add_index,
                      lattice_mesh.add_dia_mod)
        return {'FINISHED'}



class ObjectCreateMeshLattice(bpy.types.Operator):
    """Create Mesh Lattice"""
    bl_idname = "object.create_meshlattice"
    bl_label = "Create Simple Lattice"
    bl_options = {'REGISTER', 'UNDO'}
    


    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'


    def execute(self, context):
        lattice_mesh = context.scene.lattice_mesh
        lattice_cell = Cells[lattice_mesh.celltype]
        name = "Lattice " + lattice_cell.name
        nelems = np.array(lattice_mesh.nelems)
        corner = np.array(context.scene.cursor.location).reshape((1,3))
        v, e = make_nplattice(lattice_cell, lattice_mesh.cellsize, nelems, corner)
        verts, edges = nplattice2meshdata(v, e)
        bm, mesh = make_meshlattice(name, verts, edges, lattice_mesh.cellsize)
        bm.to_mesh(mesh)
        bm.free()
        
        newobj  = bpy.data.objects.new(name, mesh)

        # Link object to scene
        bpy.context.collection.objects.link(newobj)

        return {'FINISHED'}

class ObjectFilterLattice(bpy.types.Operator):
    """Remove edges with small angles Lattice"""
    bl_idname = "object.filter_lattice"
    bl_label = "Filter Lattice"
    bl_options = {'REGISTER', 'UNDO'}


    min_angle: FloatProperty(
        name="Min angle",
        description="Minimal angle between two edges",
        default=math.radians(12.0),
        subtype='ANGLE',
        min=0.0,
        max=math.radians(90.0),
        )


    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'


    def execute(self, context):
        obj = context.active_object
        depsgraph = context.evaluated_depsgraph_get()
        mesh = obj.evaluated_get(depsgraph).data
        nverts = len(mesh.vertices)
        nedges = len(mesh.edges)
        verts = np.zeros((3*nverts), dtype=np.float32)
        edges = np.zeros((2*nedges), dtype=np.int32)
        mesh.vertices.foreach_get('co', verts)
        mesh.edges.foreach_get('vertices', edges)
        verts.shape = (-1, 3)
        edges.shape = (-1, 2)

        new_edges = clean_edges(verts, edges, self.min_angle)
        new_mesh = bpy.data.meshes.new(mesh.name + " cleaned")
        new_mesh.from_pydata(verts, new_edges, [])
        new_obj = bpy.data.objects.new(obj.name + " cleaned", new_mesh)
        new_obj.matrix_world = obj.matrix_world.copy()
        context.collection.objects.link(new_obj)
        return {'FINISHED'}


    def invoke(self, context, event):
        settings = context.scene.lattice_mesh
        self.min_angle = settings.min_angle
        return self.execute(context)

class ObjectFillMeshLattice(bpy.types.Operator):
    """Fill Mesh Lattice"""
    bl_idname = "object.fill_meshlattice"
    bl_label = "Fill Mesh Lattice"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'


    def execute(self, context):
        obj = context.active_object
        lattice_mesh = context.scene.lattice_mesh
        lattice_cell = Cells[lattice_mesh.celltype]
        name = obj.name + " Lattice" + lattice_cell.name
        depsgraph = context.evaluated_depsgraph_get()
        mesh = obj.evaluated_get(depsgraph).data
        cellsize = lattice_mesh.cellsize
        ls = compute_levelset(mesh, cellsize / 2)
        tr = ls.transform
        active_bb = [list(x) for x in ls.evalActiveVoxelBoundingBox()]
        active_bb = np.array(active_bb, dtype=np.int32)
        lims = active_bb.copy()
        lims[1, :] += 1 # needed to make the upper limits inclusive
        # Make an array that covers the index space of the active voxels at half the resolution of the level set
        nelems = np.ceil((lims[1, :] - lims[0, :]) / 2.0).astype(np.int32).prod()
        pos_label = np.zeros(nelems, dtype=np.int32)
        in_pos = np.zeros((nelems, 3), dtype=np.float32)
        cell_radius = 0.5 * math.sqrt(3) * cellsize # radius of the circumscribed sphere
        acc = ls.getConstAccessor()
        ncells = 0
        for xi in range(lims[0, 0], lims[1, 0], 2):
            for yi in range(lims[0, 1], lims[1, 1], 2):
                for zi in range(lims[0, 2], lims[1, 2], 2):
                    ls_value = acc.getValue((xi, yi, zi))
                    if  ls_value < cell_radius:
                        if ls_value < -cell_radius:
                            pos_label[ncells] = 1 # interior cells
                        else:
                            pos_label[ncells] = 2 # border cells
                        in_pos[ncells, :] = np.array(tr.indexToWorld((xi-1, yi-1, zi-1)))
                        ncells += 1
        
        pos_idx = pos_label > 0
        verts, edges, border_verts, faces, report_data = fill_mesh(mesh, in_pos[pos_idx, :],
                                                                   pos_label[pos_idx], cellsize,
                                                                   lattice_cell, lattice_mesh.connect_boundary,
                                                                   lattice_mesh.move_dist,
                                                                   lattice_mesh.create_faces)
        border_attr = np.zeros(len(verts), dtype=np.float32)
        border_attr[list(border_verts)] = 1.0
        if lattice_mesh.filter_angle:
            new_edges = clean_edges(verts, edges, lattice_mesh.min_angle)
        else:
            new_edges = edges
        newmesh = bpy.data.meshes.new(name)
        newmesh.from_pydata(verts, new_edges, [])
        attr = newmesh.attributes.new('onborder','FLOAT', 'POINT')
        attr.data.foreach_set('value', border_attr)

        newobj  = bpy.data.objects.new(name, newmesh)
        newobj['Lattesh report'] = str(report_data)
        newobj.matrix_world = obj.matrix_world.copy()

        # Link object to scene
        bpy.context.collection.objects.link(newobj)

        if lattice_mesh.create_faces:
            surfverts, surfedges, surffaces = make_faces(verts, edges, border_attr, faces)
            surfname = name + ' surf'
            surfmesh = bpy.data.meshes.new(surfname)
            surfmesh.from_pydata(surfverts, surfedges, surffaces)
            surfmesh.validate(verbose=True)
            surfbm = bmesh.new()
            surfbm.from_mesh(surfmesh)
            bmesh.ops.holes_fill(surfbm, edges=surfbm.edges, sides=4)
            surfbm.to_mesh(surfmesh)
            surfbm.free()

            surfobj  = bpy.data.objects.new(surfname, surfmesh)
            surfobj.matrix_world = obj.matrix_world.copy()
            bpy.context.collection.objects.link(surfobj)


        return {'FINISHED'}


class ObjectFillVoronoiLattice(bpy.types.Operator):
    """Fill Voronoi Lattice"""
    bl_idname = "object.fill_voronoilattice"
    bl_label = "Fill Voronoi Lattice"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return have_scipy and context.active_object is not None and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        name = obj.name + "Voronoi"
        bb = np.array([[x for x in y] for y in obj.bound_box], dtype=np.float32)
        minp = bb.min(axis=0)
        maxp = bb.max(axis=0)
        dims = maxp - minp
        corner = minp
        corner = corner.reshape((1,3))

        lattice_mesh = context.scene.lattice_mesh
        v, e = make_npvoronoi(lattice_mesh.charac_length, dims, corner)
        verts, edges = nplattice2meshdata(v, e)
        bm, mesh = make_meshlattice(name, verts, edges, lattice_mesh.charac_length)
        depsgraph = context.evaluated_depsgraph_get()
        intersect(bm, obj, depsgraph)
        bm.to_mesh(mesh)
        bm.free()
        
        newobj  = bpy.data.objects.new(name, mesh)
        newobj.matrix_world = obj.matrix_world.copy()

        # Link object to scene
        bpy.context.collection.objects.link(newobj)

        return {'FINISHED'}




