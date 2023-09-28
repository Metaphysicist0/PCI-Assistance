import os
import numpy as np
import stl
import mesh

your_mesh = mesh.from_file("C:\\Users\\YTL\\OneDrive\\Desktop\\S2.stl")
volume, cog, inertia = your_mesh.get_mass_properties()
xyz = (your_mesh.max_ - your_mesh.min_)
sizel = round(xyz[0] / 10, 2)
sizew = round(xyz[1] / 10, 2)
sizeh = round(xyz[2] / 10, 2)
