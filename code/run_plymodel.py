"""
This is a test program designed to take a .ply file and its path as
input from the command line and check if it's ready for 
rendering through the PlyModel class. 
"""


import PlyModel as ply

filename = input("Filename or path to .ply file: \n")

    
instance = ply.PlyModel(filename)                # create class instance
    
instance.ReadPly()                               
instance.LooseGeometry()                         
instance.GroupIdentification()
instance.PolygonOrientation()

instance.Visualize()