#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import trimesh
import os
import json

print("Lade STEP-Datei...")
step_file_path = "../pxc_3209510_24_04_PT-2-5_3D.stp"

if os.path.exists(step_file_path):
    try:
        phoenix_mesh = trimesh.load(step_file_path)
        print(f"STEP-Datei erfolgreich geladen als: {type(phoenix_mesh)}")
        
        if hasattr(phoenix_mesh, 'dump'):
            meshes = list(phoenix_mesh.dump())
            print(f"Scene enthaelt {len(meshes)} Meshes")
            
            if meshes:
                first_mesh = meshes[0]
                print(f"Mesh mit {len(first_mesh.vertices)} Vertices und {len(first_mesh.faces)} Faces")
                
                # Konvertiere zu Open3D
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(first_mesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(first_mesh.faces)
                o3d_mesh.compute_vertex_normals()
                o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                
                print("Mesh erfolgreich konvertiert!")
                print("Starte einfache Visualisierung...")
                
                # Einfache Visualisierung ohne Interaktion
                o3d.visualization.draw_geometries([o3d_mesh],
                                                  window_name="Phoenix Contact Test",
                                                  width=800, height=600)
                
    except Exception as e:
        print(f"Fehler: {e}")
else:
    print("STEP-Datei nicht gefunden")