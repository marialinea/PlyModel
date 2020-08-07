import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from progress.bar import IncrementalBar


class PlyModel:
    """
    Parameter:
    -----------
     filename: str
       Path to the .ply file
    """

    
    def __init__(self, filename):
        self.filename = filename
        
        self.N = None               # total number of vertices
        self.M = None               # total number of polygons
        
        self.vertices = None        # spatial coordinates of all the vertices
        self.faces = None           # holds the vertex indices of all faces
     
        self.half_edges = None      # holds the directed half edges and associated polygons
        self.sorted_edges = None    # holds the sorted half edges and their associated polygon
        
        self.adjacency_mat = None   # adjacency matrix for the polygons
        self.group_num = None       # holds the number of continuous groups of polygons
        self.groups_index = None    # holds the polygon indices for each group
        self.groups_faces = None    # holds the polygon faces for each group
        self.manifold = None        # if True, the mesh contains manifold geometry
        self.more_neighbours = None # holds the polygon that have more than three neighbouring polygons
        self.non_manifold = None    # array containing the polygons that are not connected to two edges           
        self.normal_vectors = None  # holds the normal vectors of all the polygons
        
    def ReadPly(self):
        """ 
        Reads a .ply (binary or ascii) file and stores the vertex- and face-data in numpy arrays.
        """
        
        print("Reading file")
        
        sys_byteorder = ('>', '<')[sys.byteorder == 'little']
    
        ply_dtypes = dict([
            (b'int8', 'i1'),
            (b'char', 'i1'),
            (b'uint8', 'u1'),
            (b'uchar', 'b1'),
            (b'uchar', 'u1'),
            (b'int16', 'i2'),
            (b'short', 'i2'),
            (b'uint16', 'u2'),
            (b'ushort', 'u2'),
            (b'int32', 'i4'),
            (b'int', 'i4'),
            (b'uint32', 'u4'),
            (b'uint', 'u4'),
            (b'float32', 'f4'),
            (b'float', 'f4'),
            (b'float64', 'f8'),
            (b'double', 'f8')
        ])
        
        valid_formats = {'ascii': '', 'binary_big_endian': '>',
                         'binary_little_endian': '<'}
        
        
        filename = self.filename
    
        with open(filename, 'rb') as ply:
    
            
            if b'ply' not in ply.readline():
                raise ValueError('The file does not start whith the word ply')
            # get binary_little/big or ascii
            fmt = ply.readline().split()[1].decode()
            # get extension for building the numpy dtypes
            ext = valid_formats[fmt]
    
            line = []
            dtypes = defaultdict(list)
            count = 2
            points_size = None
            mesh_size = None
            has_texture = False
            
            # reading the information in the header
            while b'end_header' not in line and line != b'':
                line = ply.readline()
    
                if b'element' in line:
                    line = line.split()
                    name = line[1].decode()
                    size = int(line[2])
                    if name == "vertex":
                        points_size = size
                    elif name == "face":
                        mesh_size = size
    
                elif b'property' in line:
                    line = line.split()
                    # element mesh
                    if b'list' in line:
    
                        if b"vertex_indices" in line[-1] or b"vertex_index" in line[-1]:
                            mesh_names = ["n_points", "v1", "v2", "v3"]
                        else:
                            has_texture = True
                            mesh_names = ["n_coords"] + ["v1_u", "v1_v", "v2_u", "v2_v", "v3_u", "v3_v"]
    
                        if fmt == "ascii":
                            # the first number has different dtype than the list
                            dtypes[name].append(
                                (mesh_names[0], ply_dtypes[line[2]]))
                            # rest of the numbers have the same dtype
                            dt = ply_dtypes[line[3]]
                        else:
                            # the first number has different dtype than the list
                            dtypes[name].append(
                                (mesh_names[0], ext + ply_dtypes[line[2]]))
                            # rest of the numbers have the same dtype
                            dt = ext + ply_dtypes[line[3]]
    
                        for j in range(1, len(mesh_names)):
                            dtypes[name].append((mesh_names[j], dt))
                    else:
                        if fmt == "ascii":
                            dtypes[name].append(
                                (line[2].decode(), ply_dtypes[line[1]]))
                        else:
                            dtypes[name].append(
                                (line[2].decode(), ext + ply_dtypes[line[1]]))
                count += 1
    
            # for bin
            end_header = ply.tell()
    
        data = {}
    
        if fmt == 'ascii':
            top = count
            bottom = 0 if mesh_size is None else mesh_size
    
            names = [x[0] for x in dtypes["vertex"]]
    
            data["points"] = pd.read_csv(filename, sep=" ", header=None, engine="python",
                                         skiprows=top, skipfooter=bottom, usecols=names, names=names)
    
            for n, col in enumerate(data["points"].columns):
                data["points"][col] = data["points"][col].astype(
                    dtypes["vertex"][n][1])
    
            if mesh_size:
                top = count + points_size
    
                names = np.array([x[0] for x in dtypes["face"]])
                usecols = [1, 2, 3, 5, 6, 7, 8, 9, 10] if has_texture else [1, 2, 3]
                names = names[usecols]
    
                data["mesh"] = pd.read_csv(
                    filename, sep=" ", header=None, engine="python", skiprows=top, usecols=usecols, names=names)
    
                for n, col in enumerate(data["mesh"].columns):
                    data["mesh"][col] = data["mesh"][col].astype(
                        dtypes["face"][n + 1][1])
    
        else:
            with open(filename, 'rb') as ply:
                ply.seek(end_header)
                points_np = np.fromfile(ply, dtype=dtypes["vertex"], count=points_size)
                if ext != sys_byteorder:
                    points_np = points_np.byteswap().newbyteorder()
                data["points"] = pd.DataFrame(points_np)
                if mesh_size:
                    mesh_np = np.fromfile(ply, dtype=dtypes["face"], count=mesh_size)
                    if ext != sys_byteorder:
                        mesh_np = mesh_np.byteswap().newbyteorder()
                    data["mesh"] = pd.DataFrame(mesh_np)
                    data["mesh"].drop('n_points', axis=1, inplace=True)
    
        self.vertices = np.float64(data["points"].values[:,:3])
        self.faces = np.int64(data["mesh"].values)
        
        self.N = points_size
        self.M = mesh_size
        
        
        
        print("Reading file complete")
        print("------------------------------------------")
        
        return None
    
    def Rotate(self, theta, axis=0):
        """
        The function rotates the objects vertices along the x-, y- or z-axis with an angle theta.

        Parameters
        ----------
        theta : int or float 
            Rotation angle in radians
        axis : int, optional
            Corresponds to the axis of rotation. Axis 0 is the x-axis, axis 1 is the 
            y-axis and axis 2 is the z-axis. The default is 0.


        """
        
        epsilon = 1e-14
            
        cosine = np.cos(theta)
        sine = np.sin(theta)
         
        cosine = 0 if cosine < epsilon else cosine
        sine = 0 if sine < epsilon else sine
        
        if axis == 0:   
            rotation_mat = np.array([(1,0,0), (0, cosine, -sine), (0, sine, cosine)]).reshape((3,3))
            
        if axis == 1:
            rotation_mat = np.array([(cosine, 0, sine), (0,1,0), (-sine, 0, cosine)]).reshape((3,3))
            
        if axis == 2:
            rotation_mat = np.array([(cosine, -sine, 0), (sine, cosine, 0), (0,0,1)]).reshape((3,3))
            
        tmp = np.zeros(3)

        for i in range(self.N):
            x = self.vertices[i]
            for j in range(3):
                for k in range(3):
                    tmp[j] +=  rotation_mat[j][k] * x[k]
            self.vertices[i] = tmp  
            tmp = np.zeros(3)
        
        return None
   
    def SingleVertex(self):
        """
        Removes single/unconnected vertices, and corrects the number of vertices and the index values in the faces values. 

        """
        
        print("Searching for single vertices")
        
        singles = []                                                           # holds the indices of the single vertices
        
        for i in range(self.N):
            boolean_check = np.zeros(self.M)
        for j in range(self.M):
            if np.any(self.faces[j,:] == i):                                   # if True, element is set to 1, and the inner loop breaks
                boolean_check = 1
                break
        if not np.any(boolean_check):                                          # if all elements are False, the condition is executed. Is False if the vertex is unconnected
            singles.append(i)                   

        if not singles:
            print("No single vertices detected")
            print("------------------------------------------")
        else:

            # if it exists single vertices, have to change the indices in the face array
            
            singles.reverse()                                                  # iterating over the single vertices from highest to lowest
            
            # decrementing the indices which are greater than the single vertex index with one
            for i in singles:
                indices = np.where(self.faces > i)
                self.faces[indices] -= 1
                    
                                  
            self.N -= len(singles)
            self.vertices = np.delete(self.vertices, singles, axis=0)          # deleting the single vertices from the vertex array
            
            print("Deleted all single vertices")
            print("------------------------------------------")
        return None
    
    def UniqueVertices(self):
        """
        The function ensures that each set of xyz-coordinates are unique, i. e. vertex[i] != vertex[j] for all i,j, where i != j.
        If identical coordinates exists, the duplicated ones are deleted. 
        """
    
        bar = IncrementalBar("Checking uniqueness of the spatial coordiantes", max = self.N)
        duplicates = {}

        for i in range(self.N):
            bar.next()
            dup = []
            xyz1 = self.vertices[i]                                                     # spatial coordinates first polygon
            for j in range(i+1, self.N):
                xyz2 = self.vertices[j]                                                 # spatial coordinates second polygon
                if xyz1[0] == xyz2[0]:
                    if xyz1[1] == xyz2[1]:
                        if xyz1[2] == xyz2[2]:
                            dup.append(j)
            duplicates["{}".format(i)] = dup                                            # hash table of all the duplicated polygons
        bar.finish()

        if not duplicates:
            print("All sets of spatial coordiantes are unique")
        else:
            print("Removing duplicates")
            unique_vertices = []
            
            # verifying that the polygons with an empty list in the hash table are in fact unique
            for key in duplicates:
                if not duplicates[key]:
                    bool_check = []
                    for keyy in duplicates:
                        values = duplicates[keyy]
                        if int(key) not in values:
                            bool_check.append(1)
                        else:
                            bool_check.append(0)
                            break
                    if np.all(np.array(bool_check) == 1):
                        unique_vertices.append(int(key))
            
            
            # removing excess polygons from the hash table, i. e. if polygon j and k are duplicates of i, j and k can be removed from the table.
            for i in range(self.N):
                try:
                    if not duplicates["{}".format(i)]:                                     # if the polygon has no duplicates, it is deleted from the hash table
                        del duplicates["{}".format(i)]
                    index = duplicates["{}".format(i)][-1]                                 # if j and k are duplicates of i, j will also list k as a duplicate, and k will only be a empty list. 
                    for j in range(i+1,self.N):
                        try:
                            indices = duplicates["{}".format(j)]
                            if index in indices:
                                del duplicates["{}".format(j)]
                        except KeyError:
                            continue
                except KeyError:
                    continue
               
            
            # correcting the face- and vertice arrays
            
            # changing the indices of the duplicated polygons to the unique polygon
            for key, values in duplicates.items():
                unique_vertices.append(int(key))
                for i in range(len(values)):
                    indices = np.where(self.faces == values[i])
                    self.faces[indices] = key
                   
            unique_vertices.sort()    
              
                              
            # each duplicated vertex is now equal a constant > N        
            vertices = np.array([i for i in range(self.N)])                    
            const = np.max(self.N)*10
            for vertex in vertices:
                if vertex not in unique_vertices:
                    vertices[vertex] = const
                    
            
            # removing the duplicated vertices
            indices = np.where(vertices == const)
            vertices = np.delete(vertices, indices)
            self.vertices = self.vertices[vertices]
            
            # correcting the face indices
            indices = np.array([i for i in range(len(unique_vertices))])
            
            for i in range(len(vertices)):
                if not vertices[i] == indices[i]:
                    value = vertices[i]
                    where_value = np.where(self.faces == value)
                    self.faces[where_value] = indices[i]

            deleted = self.N - len(unique_vertices)
            
            self.N = len(unique_vertices)
     
            print("Deleted {} set(s) of excess coordinates".format(deleted))
            print("------------------------------------------")

        return None
    
    def UniquePolygons(self):
        """
        The function verifies that each polygon consists of three unique vertices, 
        i. e. that none of the row elements in the face array are equal to each other.
        """
        
        print("Checking that each polygon consists of three unique vertices")
        
        not_unique = [] 
        
        for i in range(self.M):
            if self.faces[i,0] == self.faces[i,1] or self.faces[i,1] == self.faces[i,2] or self.faces[i,0] == self.faces[i,2]:
                not_unique.append(i)
        
        
        if not not_unique:
            print("All polygons consists of three unique vertices")
            print("------------------------------------------")
        else:
            self.faces = np.delete(self.faces, not_unique, axis=0)
            self.M -= len(not_unique)
            
            print("All excess polygons deleted")
            print("------------------------------------------")            
        return None

    def HalfEdges(self):
        """
        Constructs the half-edges between the vertices, the edges are then sorted so
        that identical half-edges can be identified. 
    
        sorted_edges: array [3Mx3]
            The two first columns contains the two vertex indices that makes up the half-edge. 
            The third column holds the associated polygon.
         
        """
        
        print("Computing half edges")
        
        edges = np.zeros([self.M*3,2])    
        associated_polygon = np.zeros([self.M*3])

        numbers = [1,2,0]

        counter = 0
        for i in range(self.M):
            polygon = i
            for j in range(len(numbers)):
                edges[counter,0] = self.faces[i,j]                             # start vertex
                edges[counter,1] = self.faces[i,numbers[j]]                    # end vertex
                associated_polygon[counter] = polygon
                counter += 1
           
        self.half_edges = edges.copy() 
        self.half_edges = np.c_[self.half_edges, associated_polygon]
        self.half_edges = self.half_edges.astype("int64")
        
        edges = np.sort(edges)
        edges = np.c_[edges, associated_polygon]
        sorted_index = np.lexsort(np.fliplr(edges).T)
        self.sorted_edges = edges[sorted_index]
        self.sorted_edges = self.sorted_edges.astype("int64")
        
        print("Computing half edges complete")
        print("------------------------------------------")
        return None 
    
    def AdjacentPolygons(self):
        """
        Identifies neighbouring polygons, and stores the information in an adjacency matrix.
        
        adjacency_mat: arary[MxM]
            Undirected symmetric adjacency matrix. If polygon i and polygon j is a neighbouring
            pair, the matrix element A_ij = 1, if not A_ij = 0.
        
        """
        
        print("Identifying adjacent polygons")
        
        self.adjacency_mat = np.zeros([self.M, self.M])

        # the matrix is symmetric, only calculating the upper triangular values
        for i in range(len(self.sorted_edges)-1):
            edge1 = self.sorted_edges[i,0:2]
            edge2 = self.sorted_edges[i+1,0:2]
            if np.all(edge1==edge2):
                self.adjacency_mat[self.sorted_edges[i,-1], self.sorted_edges[i+1,-1]] = 1                         # A_ij = 1
        
        self.adjacency_mat = self.adjacency_mat + self.adjacency_mat.T - np.diag(np.diag(self.adjacency_mat))      # filling in the lower triangular values 
        
        print("Computing adjacent polygons complete")
        print("------------------------------------------")
        
        return None    
    
    def RemoveSingles(self):
        """
        The function removes single polygons. A polygon is single if its corresponding row and column vector in the 
        adjacency matrix is the null vector.
        
        The single polygons are removed from the adjacency matrix, vertice, face and half edge arrays.
        """   
        
        print("Searching for single polygons")
        
        single_poly = []
        
        for i in range(self.M):
            if not np.any(self.adjacency_mat[i,:] == 1):
                single_poly.append(i) 
        
        
        if not single_poly:
            print("No single polygons detected")
            print("------------------------------------------")
        else:
            print("Detected {} single polygon(s)".format(len(single_poly)))
            
            
            for i in range(len(single_poly)):
                rm_vertices = self.faces[single_poly[i]]              # the indices to the vertices belonging to the single polygon 
                
                   
                # deleting the vertices in the vertex array
                self.vertices = np.delete(self.vertices, rm_vertices, axis=0)                   
                 
                # deleting the faces in the faces array
                indices = np.where(self.faces == rm_vertices)
                decrement = np.sort(self.faces[indices])
                decrement = list(decrement[::-1])
                self.faces = np.delete(self.faces, indices[0][0], axis=0)
                
                for j in decrement:
                    self.faces[np.where(self.faces > j)] -= 1
                
                
                # deleting the half edges in the directed half edge array
                polygon_num = self.half_edges[np.where(self.half_edges[:,:-1] == rm_vertices[0])[0][-1]][-1]   # associated polygon number of the single polygon
                indices = np.where(self.half_edges[:,-1] == polygon_num)
                decrement = np.sort(self.half_edges[indices,0])[0]
                decrement = list(decrement[::-1])
                self.half_edges = np.delete(self.half_edges, indices, axis=0)
                
                for j in decrement:
                    self.half_edges[np.where(self.half_edges[:,:-1] > j)] -= 1
           
                decrement = np.where(self.half_edges[:,-1] > polygon_num)
                self.half_edges[decrement[0],-1] -= 1                            # decrementing the associated polygon numbers which are larger than the deleted polygon
                
                
                # deleting the half edges in the undirected half edge array
                polygon_num = self.sorted_edges[np.where(self.sorted_edges[:,:-1] == rm_vertices[0])[0][-1]][-1]   # associated polygon number of the single polygon
                indices = np.where(self.sorted_edges[:,-1] == polygon_num)
                decrement = np.sort(self.sorted_edges[indices,0])[0]
                decrement = list(decrement[::-1])
                self.sorted_edges = np.delete(self.sorted_edges, indices, axis=0)
                
                for j in decrement:
                    self.sorted_edges[np.where(self.sorted_edges[:,:-1] > j)] -= 1
           
                decrement = np.where(self.sorted_edges[:,-1] > polygon_num)
                self.sorted_edges[decrement[0],-1] -= 1                            # decrementing the associated polygon numbers which are larger than the deleted polygon
                   
                
                self.N -= 3                                               # correcting the number of vertices
                self.M -= 1                                               # correcting the number of polygons
                    
                    
                self.adjacency_mat = np.delete(self.adjacency_mat, (single_poly[i]), axis=0)  # deleting the associated row in the adjacency matrix
                self.adjacency_mat = np.delete(self.adjacency_mat, (single_poly[i]), axis=1)  # deleting the associated column in the adjacency matrix
                
            
            print("Deleted all single polygons")
            print("------------------------------------------")
                
            return None

    def FindingGroups(self):
        """
        The function determines how many continuous groups of polygons exists within the data set, based on the adjacency matrix. It iterates
        over all polygons, so that if A_ij and A_kj are neighbours, element A_ki is also part of that group. If it exists more than one group
        the function fills two dictionaries, one for the polygon number and one for the polygon vertices.
        """
        
        print("Identifying continuous groups of polygons")
    
        neighbours = np.where(np.tril(self.adjacency_mat) == 1)
        indices = list(zip(neighbours[0], neighbours[1]))
        group_vector = np.array([i for i in range(self.M)])
        
        for index in indices:
            index1 = index[0]
            index2 = index[1]
            high = max(group_vector[index1], group_vector[index2])
            low = min(group_vector[index1], group_vector[index2])
            if high != low:
                decrease = np.where(group_vector > high)
                same_value = np.where(group_vector == high)              # the indices in group_vector that has the same value as the high
                high = low                                               # sets the value correspoding to the highest value equal to the value of the lowest
                group_vector[same_value] = low                           # changing the values that are identical to the highest value
                group_vector[decrease] -= 1
              
        
        self.group_num = len(np.unique(group_vector))
        self.groups_index = {}
        self.groups_faces = {}
        
        
        # filling the groups dict with the corresponding polygons and their vertex indices
        for i in range(self.group_num):
            indices = np.where(group_vector == i)
            self.groups_index["group{}".format(i+1)] = indices 
            self.groups_faces["group{}".format(i+1)] = self.faces[indices]
            
        print("All groups identified")
        print("Number of groups: {}".format(self.group_num))
        print("------------------------------------------")    
         
        return None
    
    def Manifold(self, x_lim=[], y_lim=[], z_lim=[], multiplier=1, visualize=False):
        """
        The function runs a simple check whether the object represented in the data set is manifold; 
        if each edge is shared by exactly two faces. In the case of manifold geometry the following
        is true:
               V + F - E = 2,
        where V is the number of vertices, F is the number of faces and E is number of edges.
        
        If the model is non-manifold the method finds the number of edges that are not shared by exactly two 
        polygons and if any faces are sharing a common vertex but no edges. It also identifies any polygons 
        with more than three neighbours.
        
        
        Parameters
        ---------
            x_lim : twoelement list, optional
                Set the x-axis view limit from [low, high]
            y_lim : twoelement list, optional
                Set the y-axis view limit from [low, high].
            z_lim : twoelement list, optional
                Set the z-axis view limit from [low, high]
            multiplier : int or float, optional
                Multiplies the vertex coordinates with the factor "multiplier". The default is 1.
            visualize : bool, optional
                If True the functions plot and highlight problematic geometry. Polygons that contain 
                edges that are not connected to exactly two polygons are colored red, while the 
                non-problematic polygons are green. Polygons with more than three neighbours has 
                a ray pointing out/in of the polygon (depending on orientation). Vertices shared 
                among groups are colored magenta. The default is False.
        """
        
        print("Checking if the model is manifold")
        
        
        constant = self.N + self.M - len(self.half_edges)*0.5
        
        # checking if each half-edges is connected to exactly two faces
        
        Edges = np.unique(self.sorted_edges[:,:-1], axis=0)              # the unique elements in sorted_edges equals all of the edges
        HalfEdgeCounter = np.zeros(len(Edges))                           # each half-edge should appear twice if the model is manifold
        i = 0

        while i < len(self.sorted_edges):
            half_edge1 = self.sorted_edges[i,:-1]
            index = int(np.where((Edges == half_edge1).all(axis=1))[0])
            counter = 1                                                 # keeps track over number of times the half-edge has appeared
            if i == len(self.sorted_edges)-1:
                HalfEdgeCounter[index] = counter
                i += 1
            for j in range(i+1, len(self.sorted_edges)):                # iterating over the sorted half-edges until a non-matching edge is reached
                half_edge2 = self.sorted_edges[j,:-1]
                if np.all(half_edge1 == half_edge2):
                    counter += 1
                    if j == len(self.sorted_edges)-1:
                        HalfEdgeCounter[index] = counter
                        i += counter
                else:
                    HalfEdgeCounter[index] = counter
                    i += counter
                    break
                
        self.manifold =  True                      # initial assumption that the mesh is manifold
        
        appear_once = len(np.where(HalfEdgeCounter == 1)[0])
        appear_twice = len(np.where(HalfEdgeCounter == 2)[0])
        appear_more = len(np.where(HalfEdgeCounter > 2)[0])
        
        if appear_once > 0 or appear_more > 0:
            print("Some edges are not shared by exactly two faces")
            self.manifold = False
            
        if self.manifold and constant == 2:
            print("The model is manifold")
            print("------------------------------------------")
        else:
            print("The model is non-manifold \nIdentifying the problem(s)...")
            
            
           # identifying shared vertices among the groups
            shared_vertices = {}
        
            for i in range(self.N):
                sharing_groups = []
                for key in self.groups_faces:
                    values = self.groups_faces[key]
                    for j in range(len(values)):
                        if i in values[j]:
                            sharing_groups.append(key)
                            break
                if len(sharing_groups)>1:
                    shared_vertices["{}".format(i)] = sharing_groups
                    
            # identifying the polygons with more than three neighbours
            
            adjacency_list = defaultdict(list) 
            for i in range(self.M): 
                for j in range(self.M): 
                    if self.adjacency_mat[i][j] == 1: 
                        adjacency_list[i].append(j) 
                            
            more_neighbours = []            
            for key, values in adjacency_list.items():
                if len(values) > 3:
                    more_neighbours.append(int(key))

            self.more_neighbours = np.array(more_neighbours)
            
            # identifying the polygons that are constructed by edges that are not shared by exactly two faces
            
            NonManifold = []
            for i in range(len(Edges)):
                Edge = Edges[i]
                count = HalfEdgeCounter[i]
                if count != 2:
                    for j in range(len(self.sorted_edges)):
                        HalfEdge = self.sorted_edges[j,:-1]
                        if Edge[0] == HalfEdge[0] and Edge[1] == HalfEdge[1]: 
                            NonManifold.append(self.sorted_edges[j,-1])
            
            
            self.non_manifold = np.unique(np.array(NonManifold))
            self.non_manifold = self.non_manifold.astype("int64")
            
            if not shared_vertices:
                print("Number of half-edges appearing once: {}".format(appear_once))
                print("Number of half-edges appearing twice: {}".format(appear_twice))
                print("Number of half-edges appearing more than twice: {}".format(appear_more))
            elif self.manifold == True and constant != 2:
                for key, values in shared_vertices.items():
                    print("Vertex {} is shared between {}".format(key, values))
            else:
                print("Number of half-edges appearing once: {}".format(appear_once))
                print("Number of half-edges appearing twice: {}".format(appear_twice))
                print("Number of half-edges appearing more than twice: {}".format(appear_more))
                print("Shared vertices in the data set:")
                for key, values in shared_vertices.items():
                    print("Vertex {} is shared between {}".format(key, values))
            
            if visualize:
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                
                for i in range(self.M):
                    vtx = self.faces[i]
                    tri = art3d.Poly3DCollection([self.vertices[vtx]*multiplier])
                    if i in self.non_manifold:
                        tri.set_color((0.811, 0.149, 0.196,0.8))
                    else:
                        tri.set_color((0.188, 0.909, 0.329,0.8))
                    if i in self.more_neighbours:
                        x1, y1, z1 = self.vertices[self.faces[i][0]]
                        x2, y2, z2 = self.vertices[self.faces[i][1]]
                        x3, y3, z3 = self.vertices[self.faces[i][2]]
                        centroid = (np.array([((x1 + x2 + x3) / 3), ((y1 + y2 + y3) / 3), ((z1 + z2 + z3) / 3)]))*multiplier
                        X, Y, Z = centroid
                        U, V, W = (self.normal_vectors[i,1:]/np.linalg.norm(self.normal_vectors[i,1:]))*multiplier
                    
                        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.1)
                        
                    tri.set_edgecolor("black")
                    
                    ax.add_collection3d(tri)
                
               
                
                for key in shared_vertices:
                    x = self.vertices[int(key)][0]*multiplier
                    y = self.vertices[int(key)][1]*multiplier
                    z = self.vertices[int(key)][2]*multiplier
                    
                    ax.scatter(x,y,z, s=10**2, color="m", marker="o")
                
                if not x_lim and not y_lim and not z_lim:
                    x_lim = [np.min(self.vertices[:,0]),np.max(self.vertices[:,0])]
                    y_lim = [np.min(self.vertices[:,1]),np.max(self.vertices[:,1])]
                    z_lim = [np.min(self.vertices[:,2]),np.max(self.vertices[:,2])]
                    
                    ax.set_xlim3d(x_lim[0], x_lim[1])
                    ax.set_ylim3d(y_lim[0], y_lim[1])
                    ax.set_zlim3d(z_lim[0], z_lim[1])
                    
                else:
                    ax.set_xlim3d(x_lim[0], x_lim[1])
                    ax.set_ylim3d(y_lim[0], y_lim[1])
                    ax.set_zlim3d(z_lim[0], z_lim[1])
            
            
            
                plt.show()
            
            print("------------------------------------------")
            
            
        return None
    
    def NormalVectors(self):
        """
        The function calculates the normal vectors of the faces and stores them in an array.
        """
        print("Calculating normal vectors")
        self.normal_vectors = np.zeros([self.M, 4])
        
        for index, polygon in enumerate(self.faces):
            vec1 = self.vertices[polygon[1]] - self.vertices[polygon[0]]
            vec2 = self.vertices[polygon[2]] - self.vertices[polygon[0]]
            cross_product = np.cross(vec1,vec2)
            self.normal_vectors[index] = [index, cross_product[0], cross_product[1], cross_product[2]]
          
        print("Calculation complete")
        print("------------------------------------------")
        return None
    
    def FlippPolygons(self):
        """
        The function checks if the direction of traversal for the half-edges in each group is the same for 
        neighbouring polygons. If it is not the polygon is flipped. 
        """

        print("Checking if polygon orientation is consistent")
        
        # changing the adjacency matrix to a adjaceny list
        adjacency_list = defaultdict(list) 
        for i in range(self.M): 
            for j in range(self.M): 
                if self.adjacency_mat[i][j] == 1: 
                    adjacency_list[i].append(j) 
        
        #print(adjacency_list)     
        counter = 0
        for key, values in self.groups_index.items():
            polygons = list(values[0])
            while polygons:
                poly = polygons[0]                                      # starting polygon
                HE1 = np.where(self.half_edges[:,-1] == poly)           # indices to the half-edges belonging to the start polygon
                neighbours = adjacency_list[poly]                       # the neighbours to the start polygon
                for neighbour in neighbours:                            # iterating over the neighbours to the start polygon
                    if neighbour in polygons:                           # verifying that the polygon are not previously checked
                        polygons.remove(neighbour)                      # to make sure the polygon are not iterated over at a later time
                        polygons.insert(0, neighbour)                   # by making the neighbour the first element, its neighbours are checked in the next iteration
                        HE2 = np.where(self.half_edges[:,-1] == neighbour)  # indices to the half-edges belonging to the neighbouring polygon
                        
                        # matching half-edges
                        for i in range(3):
                            half_edge1 = self.half_edges[HE1[0][i],:-1]
                            for j in range(3):
                                half_edge2 = self.half_edges[HE2[0][j],:-1]
                                if half_edge1[0] == half_edge2[0] and half_edge1[1] == half_edge2[1]:          # if the polygons have a matching pair of half-edges they have opposite orientation
                                    self.faces[neighbour][1], self.faces[neighbour][2] = self.faces[neighbour][2], self.faces[neighbour][1]
                                    indices = np.where(self.half_edges[:,-1] == neighbour) 
                                    self.half_edges[indices[0],:-1] = np.flip(self.half_edges[indices[0],:-1], axis=1) # flips the directed half-edges
                                    counter += 1
                                    break
                            else:
                                continue
                            break
                polygons.remove(poly)                              # removing the start polygon when all the neighbours are checked    
            
                
            
        print("Flipped {} polygons".format(counter))
        
        if counter != 0:
            print("Recalculating the half-edges and normal vectors")
            self.HalfEdges()
            self.NormalVectors()
        else:
            print("------------------------------------------")
        
        return None 
    
    def RayCasting(self):
        """
        The function determines if all the polygons are facing outwards or inwards. This is done through casting a ray
        from each polygon and register each time the ray intersect another polygon. If the polygons are all facing inward
        the number of intersections are an odd number, and even if facing outwards. If facing inwards all the polygons are flipped. 
        """
        
        print("Determining inward or outward orientation of polygons")
        
        # calculating the centroids for each polygon

        centroids = np.zeros([self.M,3])

        for i in range(self.M):
            x1, y1, z1 = self.vertices[self.faces[i][0]]
            x2, y2, z2 = self.vertices[self.faces[i][1]]
            x3, y3, z3 = self.vertices[self.faces[i][2]]
            
            centroids[i] = np.array([((x1 + x2 + x3) / 3), ((y1 + y2 + y3) / 3), ((z1 + z2 + z3) / 3)]) + (0 * self.normal_vectors[i,1:]/np.linalg.norm(self.normal_vectors[i,1:]))     # adding a small displacement
        
        
        
        Recalculate = False                                         # if True after the iterations, the half-edges and normal vectors are recalculated
        intersections = {}

        for key, values in self.groups_index.items():
            intersect = np.zeros(len(values[0]))
            for i in values[0]:
                origin = centroids[i]                           # origin of the ray
                direction = self.normal_vectors[i,1:]/np.linalg.norm(self.normal_vectors[i,1:])
                for j in range(len(values[0])):
                    if i != j:
                        v0 = np.array([self.vertices[self.faces[j][0]][0], self.vertices[self.faces[j][0]][1], self.vertices[self.faces[j][0]][2]])
                        v1 = np.array([self.vertices[self.faces[j][1]][0], self.vertices[self.faces[j][1]][1], self.vertices[self.faces[j][1]][2]])
                        v2 = np.array([self.vertices[self.faces[j][2]][0], self.vertices[self.faces[j][2]][1], self.vertices[self.faces[j][2]][2]])
  
                        result = self.RayTriangleIntersect(origin, direction, v0, v1, v2)          # if an intersection occurs, the function return True
                        
                        if result:
                            intersect[i] += 1

            intersections[key] = intersect
            
           
        for key, values in intersections.items():
            if not np.any(values == 0):                                      # since the model is topological consistent, if facing outwards at least one polygon should have zero intersections 
                for j in self.groups_index[key][0]:
                    self.faces[j][1], self.faces[j][2] = self.faces[j][2], self.faces[j][1]       # flipping polygons
                    Recalculate = True
                print("Detected inward orientation for {}".format(key))
          
                          
        if Recalculate:
            print("Recalculating the half-edges and normal vectors")
            self.HalfEdges()
            self.NormalVectors()
                
            print("All polygons are flipped and has an outwards orientation")
            print("------------------------------------------")
        else:
            print("All polygons face outwards")
            print("------------------------------------------") 
        
        return None
     
    @staticmethod
    def RayTriangleIntersect(origin, direction, v0, v1, v2):
        """

        Parameters
        ----------
        origin : float
            3D cartesian coordinates to the ray's origin.
        direction : float
            Normalized direction of the ray.
        v0 : float
            3D cartesian coordinates to the first vertex belonging to the polygon that spans the plane.
        v1 : float
            3D cartesian coordinates to the second vertex belonging to the polygon that spans the plane.
        v2 : float
            3D cartesian coordinates to the third vertex belonging to the polygon that spans the plane.

        Returns
        -------
        bool
            If True an intersection occurs, if False, no intersection.

        """
        
        # calculates normal vector to the plane
        v0v1 = v1-v0
        v0v2 = v2-v0
        N = np.cross(v0v1,v0v2)
    
        eps = 1e-7
        NdotRayDirection = np.dot(N, direction)
        
    
        if NdotRayDirection > -eps and NdotRayDirection < eps:            # the ray and plane are parallel
            return False
    
    
        t = (np.dot(N,v0-origin))/NdotRayDirection
        if t < 0:                                                         # the polygon is behind the ray
            return False
    
        P = origin + t*direction

        DotProduct = np.dot(direction, P-origin)
        if DotProduct < 0:                                             # if negative, the camera direction is backward
            return False
        
        edge0 = v1-v0
        vp0 = P-v0
        C = np.cross(edge0,vp0)
        if np.dot(N,C) < 0:
            return False
    
        edge1 = v2-v1
        vp1 = P-v1
        C = np.cross(edge1,vp1)
        if np.dot(N,C) < 0:
            return False
    
        edge2 = v0-v2
        vp2 = P-v2
        C = np.cross(edge2,vp2)
        if np.dot(N,C) < 0:
            return False

        return True
    
   
    def Visualize(self, x_lim=[], y_lim=[], z_lim=[], multiplier = 1, group=None, polygons=None, normals=False):
        """
        The function plots the vertices and faces in a 3D-plot.
        Parameters
        ----------
        x_lim : twoelement list, optional
            Set the x-axis view limit from [low, high]. Default is the maximum and minimum value of all the x coordinates.
        y_lim : twoelement list, optional
            Set the y-axis view limit from [low, high]. Default is the maximum and minimum value of all the y coordinates.
        z_lim : twoelement list, optional
            Set the z-axis view limit from [low, high]. Default is the maximum and minimum value of all the z coordinates.
        multiplier : int or float
            Multiplies the vertex coordinates with the factor "multiplier". The default is 1.
        group : int or list, optional
            Specify which group(s) to visualize. Default is None, which plots all of the groups.
        polygons: int or list, optional
            Specify which polygon(s) to visualize in a different color (black). If this is specified the group wise
            visualization is not possible. Default is None.
        normals : bool, optional
            Plots normal vectors if True. Default is False

        Returns
        -------
        Produces a 3D plot of the object.

        """
        
        if group != None and polygons != None:
            raise ValueError("Both parameters 'group' and 'polygons' are specified, one has to be None.")
            
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    
        v = self.vertices*multiplier
        
        centroids = np.zeros([self.M,3])

        for i in range(self.M):
            x1, y1, z1 = self.vertices[self.faces[i][0]]
            x2, y2, z2 = self.vertices[self.faces[i][1]]
            x3, y3, z3 = self.vertices[self.faces[i][2]]
                
            centroids[i] = (np.array([((x1 + x2 + x3) / 3), ((y1 + y2 + y3) / 3), ((z1 + z2 + z3) / 3)]) + (0.01 * self.normal_vectors[i,1:]/np.linalg.norm(self.normal_vectors[i,1:])))*multiplier
            
        
        if (self.group_num == 1 and group == None and polygons == None) or (self.group_num == None and group == None and polygons == None):
            f = self.faces
            
            r = np.random.rand()
            b = np.random.rand()
            g = np.random.rand()
            color = (r, g, b)
            
            
            if normals:
                for i in range(self.M):
                    X, Y, Z = centroids[i]
                    U, V, W = (self.normal_vectors[i,1:]/np.linalg.norm(self.normal_vectors[i,1:]))*multiplier
                    
                    ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.01)
                    
                color = (r, g, b,0.2)
            pc = art3d.Poly3DCollection(v[f], facecolors=color, edgecolor=(0,0,0,0.1))
            ax.add_collection(pc)
        elif group != None and polygons == None:
            
            if isinstance(group, int):
                f = self.groups_faces["group{}".format(group)]

                r = np.random.rand()
                b = np.random.rand()
                g = np.random.rand()
                color = (r, g, b)
                
                if normals:         
                    for i in self.groups_index["group{}".format(group)]:
                        for j in i:
                            X, Y, Z = centroids[j]
                            U, V, W = (self.normal_vectors[j,1:]/np.linalg.norm(self.normal_vectors[j,1:]))*multiplier
                            ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.01)
                        
                    color = (r, g, b, 0.2)
                
                pc = art3d.Poly3DCollection(v[f], facecolors=color, edgecolor=(0,0,0,0.1))
                ax.add_collection(pc)
                
            elif isinstance(group, list):
               for i in range(len(group)):
                    f = self.groups_faces["group{}".format(group[i])]

                    r = np.random.rand()
                    b = np.random.rand()
                    g = np.random.rand()
                    color = (r, g, b)
                
                    if normals:         
                        for k in self.groups_index["group{}".format(group[i])]:
                            for j in k:
                                X, Y, Z = centroids[j]
                                U, V, W = (self.normal_vectors[j,1:]/np.linalg.norm(self.normal_vectors[j,1:]))*multiplier
                                    
                                ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.01)
                        color = (r, g, b,0.2)
                    
                    pc = art3d.Poly3DCollection(v[f], facecolors=color, edgecolor=(0,0,0,0.1))
                    ax.add_collection(pc) 
            else:
                raise TypeError("group parameter is neither list or int, but {}".format(type(group)))
        elif polygons == None:
            
            for i in range(self.group_num): 

                f = self.groups_faces["group{}".format(i+1)]

                r = np.random.rand()
                b = np.random.rand()
                g = np.random.rand()
                color = (r, g, b)
                
                pc = art3d.Poly3DCollection(v[f], facecolors=color, edgecolor=(0,0,0,0.1))
                ax.add_collection(pc)

            if normals:
                for i in range(self.M):
                    X, Y, Z = centroids[i]
                    U, V, W = (self.normal_vectors[i,1:]/np.linalg.norm(self.normal_vectors[i,1:]))*multiplier
                    
                    ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.01)
                color = (r,g,b,0.2)
                
            
        elif polygons != None and group == None:
            r = np.random.rand()
            b = np.random.rand()
            g = np.random.rand()
            color = (r, g, b, 0.2)
            if isinstance(polygons, int):

                for i in range(self.M):
                    vtx = self.faces[i]
                    tri = art3d.Poly3DCollection([self.vertices[vtx]*multiplier])
                    if i == polygons and normals:
                        tri.set_color("black")
                        X, Y, Z = centroids[i]
                        U, V, W = (self.normal_vectors[i,1:]/np.linalg.norm(self.normal_vectors[i,1:]))*multiplier
                    
                        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.01)
                    elif i == polygons and not normals:
                        tri.set_color("black")
                    else:
                        tri.set_color(color)
                    tri.set_edgecolor("cyan")
                    ax.add_collection3d(tri)

            elif isinstance(polygons, list):
                
                for i in range(self.M):
                    vtx = self.faces[i]
                    tri = art3d.Poly3DCollection([self.vertices[vtx]*multiplier])
                    if i in polygons and normals:
                        tri.set_color("black")
                        X, Y, Z = centroids[i]
                        U, V, W = (self.normal_vectors[i,1:]/np.linalg.norm(self.normal_vectors[i,1:]))*multiplier*2
                    
                        ax.quiver(X,Y,Z,U,V,W,arrow_length_ratio=0.01)
                    elif i in polygons and not normals:
                        tri.set_color("black")
                    else:
                        tri.set_color(color)
                    tri.set_edgecolor("cyan")
                    ax.add_collection3d(tri)
            
            else:
                raise TypeError("polygons parameter is neither list or int, but {}".format(type(polygons)))                
        
            
        
        if not x_lim and not y_lim and not z_lim:
            x_lim = [np.min(self.vertices[:,0]),np.max(self.vertices[:,0])]
            y_lim = [np.min(self.vertices[:,1]),np.max(self.vertices[:,1])]
            z_lim = [np.min(self.vertices[:,2]),np.max(self.vertices[:,2])]
            
            ax.set_xlim3d(x_lim[0], x_lim[1])
            ax.set_ylim3d(y_lim[0], y_lim[1])
            ax.set_zlim3d(z_lim[0], z_lim[1])
            
        else:
            ax.set_xlim3d(x_lim[0], x_lim[1])
            ax.set_ylim3d(y_lim[0], y_lim[1])
            ax.set_zlim3d(z_lim[0], z_lim[1])
            
        
        plt.show()

        return None
    

    def WriteStaticVertices(self, path=None):
        """
        The function writes the vertex number, followed by its x, y and z coordinates to a .csv file. There is 
        one line for each vertex. The file starts with a header (Static vertex), followed by number of elements in
        the file and then "end_header", before the vertex data is listed. The structure is illustrated below.
            
            Static vertex
            Number of elements
            end_header
            0,x1,y1,z1
               ...      
             
              
        Parameter
        --------
        path : str, optional
            Specifies the location to the written .csv file, excluding filename. Default is None, so the file will
            be saved to the current directory. The filename is "static_vertices".
                
        """
        
        Header = "Static vertex \n{}\nend_header \n".format(self.N)
        
        
        if path == None:
            outfilename = "static_vertices.csv"
        else:
            outfilename = path + "/" + outfilename
        
        with open(outfilename, "w") as outfile:
            outfile.write(Header)
            for index, coordinate in enumerate(self.vertices):
                outfile.write("{},{},{},{} \n".format(index, coordinate[0], coordinate[1], coordinate[2]))

        return None

    def WriteStaticPolygons(self, path=None):
        """
        The function writes the polygon number, followed by the vertex number of each of the three polygon vertices
        in right-hand oriented order to a .csv file. There is one line for each polygon. The file starts with a 
        header (Static polygon), followed by number of elements in the file and then "end_header", before the 
        polygon data is listed. The structure is illustrated below.
            
            Static polygon
            Number of elements
            end_header
            0,v1,v2,v3
               ...      
        
        Parameter
        --------
        path : str, optional
            Specifies the location to the written .csv file, excluding filename. Default is None, so the file will
            be saved to the current directory. The filename is "static_polygons".
                
        """
        
        Header = "Static polygon \n{}\nend_header \n".format(self.M)
        
        
        if path == None:
            outfilename = "static_polygons.csv"
        else:
            outfilename = path + "/" + outfilename
        
        with open(outfilename, "w") as outfile:
            outfile.write(Header)
            for index, vertices in enumerate(self.faces):
                outfile.write("{},{},{},{} \n".format(index, vertices[0], vertices[1], vertices[2]))

        return None
    
    def LooseGeometry(self):
        """
        A collection of class methods that ensures that the model
        contains no loose geometry.

        """
        self.SingleVertex()
        self.UniqueVertices()
        self.UniquePolygons()
        
        return None
   
    def GroupIdentification(self):
        """
        The methods listed below contribute to identify the 
        continuous groups in the data set.
        """
        self.HalfEdges()
        self.AdjacentPolygons()
        self.RemoveSingles()
        self.FindingGroups()
        
        return None   
    
    def PolygonOrientation(self, vis=False):
        """
        The assortment of methods determine whether the geometry is manifold or not,
        and in the case of manifold geometry ensures that the polygons are all faced 
        outward.

        Parameters
        ----------
        vis : bool, optional
            Corresponds with the visualize parameter belonging to the method 
            PlyModel.Manifold. If not specified, the default value is False.

        """
        
        self.NormalVectors()
        self.Manifold(visualize=vis)
        
        if self.manifold:
            self.FlippPolygons()
            self.RayCasting()
        else:
            print("The mesh needs to be manifold before polygon orientation can be made consistent.")
        return None
    