#!/usr/bin/env python3
"""
STEP zu 3D-Pixel (Voxel) Konverter

Liest STEP-Dateien und JSON-Anschlusspunkte, konvertiert das Mesh zu Voxeln
und visualisiert das Ergebnis mit Open3D.
"""

import os
import json
import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_step_file(step_file_path):
    """Lädt eine STEP-Datei und gibt das Mesh zurück"""
    print(f"Lade STEP-Datei: {step_file_path}")
    
    if not os.path.exists(step_file_path):
        raise FileNotFoundError(f"STEP-Datei nicht gefunden: {step_file_path}")
    
    # Lade mit trimesh
    mesh = trimesh.load(step_file_path)
    
    # Handle unterschiedliche trimesh Rückgabetypen
    if hasattr(mesh, 'dump'):
        # Scene mit mehreren Meshes
        meshes = list(mesh.dump())
        if meshes:
            mesh = meshes[0]  # Nehme das erste Mesh
        else:
            raise ValueError("Keine Meshes in der STEP-Datei gefunden")
    
    print(f"Mesh geladen: {len(mesh.vertices)} Vertices, {len(mesh.faces)} Faces")
    return mesh

def load_json_vectors(json_file_path):
    """Lädt Anschlussvektoren aus JSON-Datei"""
    if not os.path.exists(json_file_path):
        print(f"Warnung: JSON-Datei nicht gefunden: {json_file_path}")
        return []
    
    print(f"Lade JSON-Vektoren: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vectors = data.get('connection_vectors', [])
    print(f"JSON geladen: {len(vectors)} Anschlussvektoren")
    
    return vectors

def mesh_to_voxels(mesh, voxel_resolution=50):
    """Konvertiert trimesh Mesh zu Open3D Voxel Grid"""
    print(f"Konvertiere Mesh zu Voxeln (Resolution: {voxel_resolution})")
    
    # Konvertiere trimesh zu Open3D Mesh
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.faces)
    
    # Erstelle Open3D Mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Berechne Normalen für bessere Voxelisierung
    o3d_mesh.compute_vertex_normals()
    
    # Konvertiere zu Voxel Grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        o3d_mesh, 
        voxel_size=calculate_voxel_size(mesh, voxel_resolution)
    )
    
    print(f"Voxel Grid erstellt: {len(voxel_grid.get_voxels())} Voxel")
    return voxel_grid

def calculate_voxel_size(mesh, resolution):
    """Berechnet optimale Voxel-Größe basierend auf Mesh-Dimensionen"""
    # Berechne Bounding Box
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]  # max - min
    max_dimension = np.max(dimensions)
    
    # Voxel-Größe so wählen, dass wir die gewünschte Resolution erreichen
    voxel_size = max_dimension / resolution
    print(f"Berechnete Voxel-Größe: {voxel_size:.6f}")
    
    return voxel_size

def create_vector_markers(vectors):
    """Erstellt Open3D Geometrien für die Anschlussvektoren"""
    geometries = []
    
    for i, vector in enumerate(vectors):
        pos = vector['position']
        dir_vec = vector['direction']
        
        # Erstelle Kugel am Anschlusspunkt
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        sphere.translate([pos['x'], pos['y'], pos['z']])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Rot
        geometries.append(sphere)
        
        # Erstelle Pfeil für Richtung
        arrow_length = 0.01
        end_point = [
            pos['x'] + dir_vec['x'] * arrow_length,
            pos['y'] + dir_vec['y'] * arrow_length,
            pos['z'] + dir_vec['z'] * arrow_length
        ]
        
        # Linie für Pfeil
        line_points = [[pos['x'], pos['y'], pos['z']], end_point]
        line_indices = [[0, 1]]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # Rot
        geometries.append(line_set)
    
    print(f"Erstellt: {len(geometries)} Vektor-Marker")
    return geometries

def voxels_to_depth_image(voxel_grid, direction='z'):
    """
    Erstellt ein vollständiges Tiefenbild aus einem Voxel Grid
    
    Args:
        voxel_grid: Open3D VoxelGrid
        direction: 'x', 'y', oder 'z' - Projektionsrichtung
    
    Returns:
        depth_image: 2D numpy array mit Tiefenwerten
        extent: [x_min, x_max, y_min, y_max] für korrekte Darstellung
    """
    print(f"Erstelle Tiefenbild in {direction.upper()}-Richtung...")
    
    # Hole alle Voxel
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        print("Warnung: Keine Voxel gefunden!")
        return None, None
    
    # Extrahiere Voxel-Koordinaten und konvertiere zu Weltkoordinaten
    voxel_coords = []
    for voxel in voxels:
        # Voxel Grid Koordinaten zu Weltkoordinaten
        world_coord = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        voxel_coords.append(world_coord)
    
    voxel_coords = np.array(voxel_coords)
    
    if direction.lower() == 'z':
        # Projektion in Z-Richtung (Draufsicht)
        x_coords = voxel_coords[:, 0]
        y_coords = voxel_coords[:, 1] 
        depth_coords = voxel_coords[:, 2]
        
        # Bestimme Bild-Auflösung (niedriger für vollständigeres Bild)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Verwende Voxel-Auflösung direkt - jeder Voxel wird ein Pixel
        voxel_size = voxel_grid.voxel_size
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Auflösung entspricht der Anzahl der einzigartigen Voxel-Positionen
        resolution_x = int(np.ceil(x_range / voxel_size)) + 1
        resolution_y = int(np.ceil(y_range / voxel_size)) + 1
        
        print(f"Tiefenbild-Auflösung: {resolution_x} x {resolution_y} (1 Pixel = 1 Voxel)")
        
        # Erstelle Tiefenbild mit exakter Voxel-Zuordnung
        depth_image = np.full((resolution_y, resolution_x), np.nan)
        
        # Konvertiere jeder Voxel-Position zu Pixel-Koordinaten
        for i in range(len(voxel_coords)):
            x_pos = x_coords[i]
            y_pos = y_coords[i]
            depth_value = depth_coords[i]
            
            # Berechne Pixel-Koordinaten direkt aus Voxel-Position
            x_idx = int(round((x_pos - x_min) / voxel_size))
            y_idx = int(round((y_pos - y_min) / voxel_size))
            
            # Prüfe Grenzen
            if 0 <= x_idx < resolution_x and 0 <= y_idx < resolution_y:
                current_depth = depth_image[y_idx, x_idx]
                
                # Speichere maximale Tiefe (am weitesten vom Betrachter entfernt)  
                if np.isnan(current_depth) or depth_value > current_depth:
                    depth_image[y_idx, x_idx] = depth_value
        
        extent = [x_min, x_max, y_min, y_max]
        
    else:
        print(f"Richtung '{direction}' noch nicht implementiert")
        return None, None
    
    # Keine Interpolation - zeige nur direkte Voxel-Punkte
    
    # Statistiken
    valid_pixels = ~np.isnan(depth_image)
    num_valid = np.sum(valid_pixels)
    total_pixels = depth_image.size
    coverage = num_valid / total_pixels * 100
    
    print(f"Tiefenbild erstellt: {depth_image.shape}, {num_valid}/{total_pixels} Pixel ({coverage:.1f}% Abdeckung)")
    
    if num_valid > 0:
        depth_min, depth_max = np.nanmin(depth_image), np.nanmax(depth_image)
        print(f"Tiefenbereich: {depth_min:.6f} bis {depth_max:.6f}")
    
    return depth_image, extent

def interpolate_depth_gaps(depth_image):
    """Füllt kleine Lücken im Tiefenbild durch Interpolation"""
    from scipy import ndimage
    
    # Finde gültige Pixel
    valid_mask = ~np.isnan(depth_image)
    
    if np.sum(valid_mask) < 10:  # Zu wenig Daten für Interpolation
        return depth_image
    
    # Erstelle eine Kopie für Interpolation
    filled_image = depth_image.copy()
    
    # Einfache Nahbereichs-Interpolation für kleine Lücken
    kernel = np.ones((3,3))
    
    for iteration in range(3):  # Mehrere Iterationen für größere Lücken
        # Finde Pixel die interpoliert werden können (haben gültige Nachbarn)
        valid_neighbors = ndimage.convolve(valid_mask.astype(float), kernel, mode='constant', cval=0)
        can_interpolate = (valid_neighbors > 0) & np.isnan(filled_image)
        
        if not np.any(can_interpolate):
            break
            
        # Interpoliere diese Pixel
        for y, x in np.argwhere(can_interpolate):
            # Hole gültige Nachbarwerte in 3x3 Umgebung
            y_start, y_end = max(0, y-1), min(filled_image.shape[0], y+2)
            x_start, x_end = max(0, x-1), min(filled_image.shape[1], x+2)
            
            neighbors = filled_image[y_start:y_end, x_start:x_end]
            valid_neighbors = ~np.isnan(neighbors)
            
            if np.any(valid_neighbors):
                # Verwende Mittelwert der gültigen Nachbarn
                filled_image[y, x] = np.nanmean(neighbors[valid_neighbors])
        
        # Update valid mask
        valid_mask = ~np.isnan(filled_image)
    
    return filled_image

def visualize_depth_image(depth_image, extent, title="Tiefenbild"):
    """Visualisiert das Tiefenbild mit matplotlib"""
    if depth_image is None:
        print("Kein Tiefenbild zu visualisieren")
        return
        
    plt.figure(figsize=(10, 8))
    
    # Verwende eine Farbskala die NaN-Werte transparent macht
    cmap = plt.cm.viridis
    cmap.set_bad(color='white', alpha=0.5)  # NaN-Werte in weiß/transparent
    
    plt.imshow(depth_image, 
               extent=extent, 
               origin='lower', 
               cmap=cmap,
               interpolation='nearest')
    
    plt.colorbar(label='Tiefe (Z-Koordinate)')
    plt.title(title)
    plt.xlabel('X-Koordinate')
    plt.ylabel('Y-Koordinate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_depth_image(depth_image, filename="depth_image.png"):
    """Speichert das Tiefenbild als Bild-Datei"""
    if depth_image is None:
        print("Kein Tiefenbild zu speichern")
        return
        
    # Normalisiere Tiefenwerte zu 0-255 für Bildexport
    valid_mask = ~np.isnan(depth_image)
    if np.sum(valid_mask) == 0:
        print("Keine gültigen Tiefenwerte zum Speichern")
        return
        
    # Normalisiere nur gültige Werte
    depth_normalized = np.zeros_like(depth_image)
    valid_depths = depth_image[valid_mask]
    depth_min, depth_max = valid_depths.min(), valid_depths.max()
    
    if depth_max > depth_min:
        depth_normalized[valid_mask] = 255 * (valid_depths - depth_min) / (depth_max - depth_min)
    
    # Setze ungültige Pixel auf 0 (schwarz)
    depth_normalized[~valid_mask] = 0
    
    # Speichere als PNG
    plt.imsave(filename, depth_normalized.astype(np.uint8), cmap='viridis')
    print(f"Tiefenbild gespeichert: {filename}")

def visualize_voxels_and_vectors(voxel_grid, vector_geometries):
    """Visualisiert Voxel Grid und Anschlussvektoren mit Open3D"""
    print("Starte 3D-Visualisierung...")
    
    # Sammle alle Geometrien
    geometries = [voxel_grid] + vector_geometries
    
    # Erstelle Koordinatensystem
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.02, origin=[0, 0, 0]
    )
    geometries.append(coordinate_frame)
    
    # Visualisiere
    o3d.visualization.draw_geometries(
        geometries,
        window_name="STEP 3D-Pixel Viewer",
        width=1200,
        height=800
    )

def main():
    """Hauptfunktion"""
    import sys
    
    # Prüfe Command Line Arguments
    if len(sys.argv) > 1:
        step_file = sys.argv[1]
    else:
        # Fallback auf festen Pfad
        step_file = "pxc_3209510_24_04_PT-2-5_3D.stp"
    
    # Bestimme JSON-Dateiname basierend auf STEP-Datei
    step_name = os.path.splitext(step_file)[0]
    json_file = f"{step_name}.json"
    
    try:
        # 1. Lade STEP-Datei
        mesh = load_step_file(step_file)
        
        # 2. Lade JSON-Vektoren
        vectors = load_json_vectors(json_file)
        
        # 3. Konvertiere zu Voxeln
        voxel_grid = mesh_to_voxels(mesh, voxel_resolution=100)
        
        # 4. Erstelle Vektor-Marker
        vector_geometries = create_vector_markers(vectors)
        
        # 5. Erstelle Tiefenbild
        depth_image, extent = voxels_to_depth_image(voxel_grid, direction='z')
        
        # 6. Visualisiere Tiefenbild
        if depth_image is not None:
            visualize_depth_image(depth_image, extent, "Tiefenbild (Z-Projektion)")
            
            # Optional: Speichere Tiefenbild
            output_name = f"{step_name}_depth_image.png"
            save_depth_image(depth_image, output_name)
        
        # 3D-Voxel-Visualisierung übersprungen (nicht mehr benötigt)
        
        print("Programm beendet.")
        
    except Exception as e:
        print(f"Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()