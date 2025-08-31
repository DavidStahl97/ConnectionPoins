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
import cv2

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

def differentiate_depth_image(depth_image):
    """Berechnet Gradient/Differenzierung des Tiefenbilds mit OpenCV"""
    if depth_image is None:
        print("Kein Tiefenbild für Differenzierung")
        return None, None, None
    
    print("Berechne Tiefenbild-Differenzierung...")
    
    # Konvertiere NaN-Werte zu 0 für OpenCV
    valid_mask = ~np.isnan(depth_image)
    depth_clean = np.where(valid_mask, depth_image, 0)
    
    # Konvertiere zu 8-bit für OpenCV (normalisiert)
    if np.sum(valid_mask) > 0:
        valid_depths = depth_image[valid_mask]
        depth_min, depth_max = valid_depths.min(), valid_depths.max()
        
        if depth_max > depth_min:
            depth_8bit = np.zeros_like(depth_clean, dtype=np.uint8)
            depth_8bit[valid_mask] = 255 * (valid_depths - depth_min) / (depth_max - depth_min)
        else:
            depth_8bit = np.zeros_like(depth_clean, dtype=np.uint8)
    else:
        depth_8bit = np.zeros_like(depth_clean, dtype=np.uint8)
    
    # Berechne Gradienten mit Sobel-Operatoren
    grad_x = cv2.Sobel(depth_8bit, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_8bit, cv2.CV_64F, 0, 1, ksize=3)
    
    # Berechne Gradientenmagnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Setze Gradienten ungültiger Bereiche auf 0
    grad_x[~valid_mask] = 0
    grad_y[~valid_mask] = 0
    gradient_magnitude[~valid_mask] = 0
    
    print(f"Differenzierung berechnet - Max Gradient: {gradient_magnitude.max():.2f}")
    
    return grad_x, grad_y, gradient_magnitude

def detect_contours(gradient_magnitude, threshold_factor=0.1):
    """
    Erkennt geschlossene Konturen im Gradientenmagnitude-Bild
    
    Args:
        gradient_magnitude: 2D numpy array mit Gradientenmagnitude-Werten
        threshold_factor: Schwellenwert-Faktor (0.0-1.0) für Konturenerkennung
    
    Returns:
        contours: Liste der erkannten Konturen
        binary_image: Binäres Bild für Konturenerkennung
        contour_image: RGB-Bild mit gezeichneten Konturen
    """
    print("Erkenne geschlossene Konturen...")
    
    if gradient_magnitude is None:
        return [], None, None
    
    # Konvertiere zu 8-bit für OpenCV
    valid_mask = gradient_magnitude > 0
    if not np.any(valid_mask):
        print("Warnung: Keine gültigen Gradientenwerte für Konturenerkennung")
        return [], None, None
    
    # Normalisiere Gradienten zu 0-255
    grad_normalized = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    valid_values = gradient_magnitude[valid_mask]
    grad_min, grad_max = valid_values.min(), valid_values.max()
    
    if grad_max > grad_min:
        grad_normalized[valid_mask] = 255 * (valid_values - grad_min) / (grad_max - grad_min)
    
    # Schwellenwert für Binärbild
    threshold_value = int(255 * threshold_factor)
    _, binary_image = cv2.threshold(grad_normalized, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morphologische Operationen zur Verbesserung der Konturen
    kernel = np.ones((2,2), np.uint8)  # Kleinerer Kernel für feine Linien
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)  # Open kann dünne Linien zerstören
    
    # Finde Konturen mit Hierarchie-Information
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtere kleine Konturen heraus
    min_contour_area = 20  # Minimale Konturengröße in Pixeln (reduziert für Randlinien)
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    
    print(f"Konturen gefunden: {len(contours)} total, {len(filtered_contours)} nach Filterung")
    
    # Erstelle Farbvisualisierung der Konturen
    contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    
    # Zeichne Konturen in verschiedenen Farben
    colors = [
        (255, 0, 0),    # Rot
        (0, 255, 0),    # Grün  
        (0, 0, 255),    # Blau
        (255, 255, 0),  # Gelb
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Violett
    ]
    
    for i, contour in enumerate(filtered_contours):
        color = colors[i % len(colors)]
        cv2.drawContours(contour_image, [contour], -1, color, 2)
        
        # Füge Kontur-Information hinzu
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Berechne Zentroid für Text-Position
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(contour_image, f"C{i+1}", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return filtered_contours, binary_image, contour_image, hierarchy

def filter_nested_contours_with_hierarchy(contours, hierarchy):
    """
    Entfernt verschachtelte Konturen basierend auf OpenCV Hierarchie-Information
    
    Args:
        contours: Liste von OpenCV Konturen
        hierarchy: OpenCV Hierarchie-Array [next, previous, first_child, parent]
    
    Returns:
        filtered_contours: Liste von Konturen ohne verschachtelte Konturen
    """
    if len(contours) <= 1 or hierarchy is None:
        return contours
    
    print("Filtere verschachtelte Konturen mit OpenCV Hierarchie...")
    
    # Hierarchie-Format: [next, previous, first_child, parent]
    # parent == -1: Kontur ist auf oberster Ebene (nicht verschachtelt)
    # parent >= 0: Kontur hat eine Eltern-Kontur (ist verschachtelt)
    
    filtered_indices = []
    removed_count = 0
    
    for i in range(len(contours)):
        parent_index = hierarchy[0][i][3]  # Parent-Index
        
        if parent_index == -1:
            # Kontur ist auf oberster Ebene - behalten
            filtered_indices.append(i)
        else:
            # Kontur ist verschachtelt - entfernen
            print(f"  Entferne Kontur C{i+1} (Kind von C{parent_index+1})")
            removed_count += 1
    
    filtered_contours = [contours[i] for i in filtered_indices]
    
    print(f"Verschachtelte Konturen: {len(contours)} -> {len(filtered_contours)} (entfernt: {removed_count})")
    
    return filtered_contours

def complete_individual_cut_contours(binary_image, contours, frame_width=3):
    """
    Vervollständigt einzelne abgeschnittene Konturen durch Hinzufügen eines Rahmens
    und Verbindung nur derjenigen Randpunkte, die zur selben Kontur gehören
    
    Args:
        binary_image: Ursprüngliches Binärbild
        contours: Liste der erkannten Konturen
        frame_width: Breite des hinzuzufügenden Rahmens
    
    Returns:
        completed_image: Erweitertes Binärbild mit vervollständigten Konturen
        new_contours: Neu erkannte Konturen im erweiterten Bild
    """
    print(f"Vervollständige einzelne abgeschnittene Konturen (Rahmen: {frame_width} Pixel)...")
    
    old_height, old_width = binary_image.shape
    
    # Erstelle erweitertes Bild mit Rahmen
    new_height = old_height + 2 * frame_width
    new_width = old_width + 2 * frame_width
    
    # Neues Bild mit schwarzem Rahmen erstellen
    completed_image = np.zeros((new_height, new_width), dtype=np.uint8)
    
    # Ursprüngliches Bild in die Mitte setzen
    completed_image[frame_width:frame_width + old_height, 
                   frame_width:frame_width + old_width] = binary_image
    
    connections_made = 0
    completed_contours = 0
    
    # Prüfe jede Kontur einzeln auf Randabschnitte
    for contour_idx, contour in enumerate(contours):
        # Finde alle Randpunkte dieser spezifischen Kontur
        edge_points_per_side = {
            'oben': [],     # y = 0
            'unten': [],    # y = old_height - 1  
            'links': [],    # x = 0
            'rechts': []    # x = old_width - 1
        }
        
        for point in contour:
            x, y = point[0]  # OpenCV Kontur-Format
            
            # Prüfe ob Punkt am ursprünglichen Rand lag
            if y == 0:  # Oberer Rand
                edge_points_per_side['oben'].append((x + frame_width, y + frame_width))
            elif y == old_height - 1:  # Unterer Rand  
                edge_points_per_side['unten'].append((x + frame_width, y + frame_width))
                
            if x == 0:  # Linker Rand
                edge_points_per_side['links'].append((x + frame_width, y + frame_width))
            elif x == old_width - 1:  # Rechter Rand
                edge_points_per_side['rechts'].append((x + frame_width, y + frame_width))
        
        # Vervollständige nur diese Kontur, wenn sie Randpunkte hat
        contour_was_completed = False
        
        for edge_name, points in edge_points_per_side.items():
            if len(points) >= 2:
                print(f"  Kontur C{contour_idx+1}: Vervollständige {len(points)} Punkte am {edge_name}en Rand")
                contour_was_completed = True
                
                if edge_name == 'oben':
                    # Verbinde über die oberste Rahmenlinie
                    y_line = frame_width - 1
                    points.sort()  # Nach X sortieren
                    x1, _ = points[0]   # Erster Punkt
                    x2, _ = points[-1]  # Letzter Punkt
                    
                    # Verbindung nur zwischen erstem und letztem Punkt dieser Kontur
                    completed_image[y_line, x1:x2+1] = 255
                    # Verbindungslinien zu den ursprünglichen Punkten
                    completed_image[y_line:y_line + frame_width + 1, x1] = 255
                    completed_image[y_line:y_line + frame_width + 1, x2] = 255
                    connections_made += 1
                        
                elif edge_name == 'unten':
                    # Verbinde über die unterste Rahmenlinie
                    y_line = new_height - frame_width
                    points.sort()  # Nach X sortieren
                    x1, _ = points[0]   # Erster Punkt
                    x2, _ = points[-1]  # Letzter Punkt
                    
                    # Verbindung nur zwischen erstem und letztem Punkt dieser Kontur
                    completed_image[y_line, x1:x2+1] = 255
                    # Verbindungslinien zu den ursprünglichen Punkten
                    completed_image[y_line - frame_width:y_line + 1, x1] = 255  
                    completed_image[y_line - frame_width:y_line + 1, x2] = 255
                    connections_made += 1
                        
                elif edge_name == 'links':
                    # Verbinde über die linke Rahmenlinie
                    x_line = frame_width - 1
                    points.sort(key=lambda p: p[1])  # Nach Y sortieren
                    _, y1 = points[0]   # Erster Punkt
                    _, y2 = points[-1]  # Letzter Punkt
                    
                    # Verbindung nur zwischen erstem und letztem Punkt dieser Kontur
                    completed_image[y1:y2+1, x_line] = 255
                    # Verbindungslinien zu den ursprünglichen Punkten
                    completed_image[y1, x_line:x_line + frame_width + 1] = 255
                    completed_image[y2, x_line:x_line + frame_width + 1] = 255
                    connections_made += 1
                        
                elif edge_name == 'rechts':
                    # Verbinde über die rechte Rahmenlinie  
                    x_line = new_width - frame_width
                    points.sort(key=lambda p: p[1])  # Nach Y sortieren
                    _, y1 = points[0]   # Erster Punkt
                    _, y2 = points[-1]  # Letzter Punkt
                    
                    # Verbindung nur zwischen erstem und letztem Punkt dieser Kontur
                    completed_image[y1:y2+1, x_line] = 255
                    # Verbindungslinien zu den ursprünglichen Punkten
                    completed_image[y1, x_line - frame_width:x_line + 1] = 255
                    completed_image[y2, x_line - frame_width:x_line + 1] = 255
                    connections_made += 1
        
        if contour_was_completed:
            completed_contours += 1
    
    # Erkenne neue Konturen im erweiterten Bild mit Hierarchie
    new_contours, new_hierarchy = cv2.findContours(completed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Kontur-Vervollständigung: {completed_contours} Konturen vervollständigt, {connections_made} Verbindungen")
    print(f"Bild erweitert: {old_width}x{old_height} -> {new_width}x{new_height}")
    print(f"Neue Konturen: {len(new_contours)} (vorher: {len(contours)})")
    
    return completed_image, new_contours, new_hierarchy

def save_contour_analysis(binary_image, contour_image, contours, hierarchy, extent, step_name, vectors=None, output_dir=None):
    """Speichert die Kontur-Analyse als Bilder"""
    print("Speichere Kontur-Analyse...")
    
    # Vervollständige einzelne abgeschnittene Konturen mit Rahmen
    completed_image, completed_contours, completed_hierarchy = complete_individual_cut_contours(binary_image, contours, frame_width=5)
    
    # Filtere verschachtelte Konturen mit OpenCV Hierarchie (verwende erweiterte Konturen)
    filtered_contours = filter_nested_contours_with_hierarchy(completed_contours, completed_hierarchy)
    
    # Erstelle gefiltertes Konturen-Bild basierend auf vervollständigtem Bild
    filtered_contour_image = cv2.cvtColor(completed_image, cv2.COLOR_GRAY2RGB)
    
    colors = [
        (255, 0, 0),    # Rot
        (0, 255, 0),    # Grün  
        (0, 0, 255),    # Blau
        (255, 255, 0),  # Gelb
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Violett
    ]
    
    for i, contour in enumerate(filtered_contours):
        color = colors[i % len(colors)]
        cv2.drawContours(filtered_contour_image, [contour], -1, color, 2)
        
        # Berechne Zentroid für Text-Position
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(filtered_contour_image, f"F{i+1}", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Bestimme Dateinamen
    if output_dir:
        binary_filename = os.path.join(output_dir, f"{step_name}_contours_binary.png")
        contour_filename = os.path.join(output_dir, f"{step_name}_contours_detected.png")
        filtered_filename = os.path.join(output_dir, f"{step_name}_contours_filtered.png")
        combined_contour_filename = os.path.join(output_dir, f"{step_name}_contours_analysis.png")
    else:
        binary_filename = f"{step_name}_contours_binary.png"
        contour_filename = f"{step_name}_contours_detected.png"
        filtered_filename = f"{step_name}_contours_filtered.png"
        combined_contour_filename = f"{step_name}_contours_analysis.png"
    
    # Speichere Binärbild
    cv2.imwrite(binary_filename, binary_image)
    print(f"Binärbild gespeichert: {binary_filename}")
    
    # Speichere Konturen-Bild  
    cv2.imwrite(contour_filename, cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))
    print(f"Konturen-Bild gespeichert: {contour_filename}")
    
    # Speichere gefiltertes Konturen-Bild
    cv2.imwrite(filtered_filename, cv2.cvtColor(filtered_contour_image, cv2.COLOR_RGB2BGR))
    print(f"Gefiltertes Konturen-Bild gespeichert: {filtered_filename}")
    
    # Berechne neues Extent für erweiterte Bilder
    old_height, old_width = binary_image.shape
    new_height, new_width = completed_image.shape
    frame_width = (new_width - old_width) // 2
    
    # Erweitere Extent entsprechend
    x_min, x_max, y_min, y_max = extent
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Berechne Pixel-zu-Koordinaten-Verhältnis
    x_scale = x_range / old_width
    y_scale = y_range / old_height
    
    # Neues Extent mit Rahmen
    new_extent = [
        x_min - frame_width * x_scale,  # x_min
        x_max + frame_width * x_scale,  # x_max  
        y_min - frame_width * y_scale,  # y_min
        y_max + frame_width * y_scale   # y_max
    ]
    
    # Erstelle kombinierte 3-Panel Visualisierung
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # Ursprüngliches Binärbild links
    axes[0].imshow(binary_image, extent=extent, origin='lower', cmap='gray', interpolation='nearest')
    axes[0].set_title('Ursprüngliches Binärbild')
    axes[0].set_xlabel('X-Koordinate')
    axes[0].set_ylabel('Y-Koordinate')
    
    # Anschlusspunkte im Binärbild
    if vectors:
        for vector in vectors:
            pos = vector['position']
            axes[0].plot(pos['x'], pos['y'], 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1)
            axes[0].text(pos['x'], pos['y'], f"  P{vector['id']}", color='red', fontweight='bold', fontsize=8)
    
    # Alle Konturen-Bild mitte (original)
    axes[1].imshow(contour_image, extent=extent, origin='lower', interpolation='nearest')
    axes[1].set_title(f'Ursprüngliche Konturen ({len(contours)})')
    axes[1].set_xlabel('X-Koordinate')  
    axes[1].set_ylabel('Y-Koordinate')
    
    # Anschlusspunkte im Konturen-Bild
    if vectors:
        for vector in vectors:
            pos = vector['position']
            axes[1].plot(pos['x'], pos['y'], 'yo', markersize=5, markeredgecolor='black', markeredgewidth=1)
            axes[1].text(pos['x'], pos['y'], f"  P{vector['id']}", color='yellow', fontweight='bold', fontsize=8)
    
    # Erweiterte und gefilterte Konturen-Bild rechts
    axes[2].imshow(filtered_contour_image, extent=new_extent, origin='lower', interpolation='nearest')
    axes[2].set_title(f'Erweitert & Gefiltert ({len(filtered_contours)} final)')
    axes[2].set_xlabel('X-Koordinate')  
    axes[2].set_ylabel('Y-Koordinate')
    
    # Anschlusspunkte im erweiterten Konturen-Bild
    if vectors:
        for vector in vectors:
            pos = vector['position']
            axes[2].plot(pos['x'], pos['y'], 'yo', markersize=5, markeredgecolor='black', markeredgewidth=1)
            axes[2].text(pos['x'], pos['y'], f"  P{vector['id']}", color='yellow', fontweight='bold', fontsize=8)
    
    # Kontur-Statistiken als Text
    stats_text = f"Alle Konturen: {len(contours)}\n"
    stats_text += f"Gefilterte Konturen: {len(filtered_contours)}\n"
    stats_text += f"Entfernt (verschachtelt): {len(contours) - len(filtered_contours)}\n\n"
    
    if filtered_contours:
        stats_text += "Finale Konturen:\n"
        for i, contour in enumerate(filtered_contours[:5]):  # Zeige nur erste 5 gefilterte Konturen
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            stats_text += f"F{i+1}: Area={area:.0f}, Umfang={perimeter:.1f}\n"
        
        if len(filtered_contours) > 5:
            stats_text += f"... und {len(filtered_contours)-5} weitere"
    else:
        stats_text += "Keine finalen Konturen"
            
    plt.figtext(0.02, 0.02, stats_text, fontsize=8, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Platz für Statistiken
    plt.savefig(combined_contour_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Kombinierte Kontur-Analyse gespeichert: {combined_contour_filename}")
    
    # Detaillierte Kontur-Informationen ausgeben
    print("Alle Konturen:")
    if contours:
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Approximiere Kontur für Formerkennung
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            print(f"  C{i+1}: Area={area:.0f}, Umfang={perimeter:.1f}, Eckpunkte={len(approx)}")
    else:
        print("  Keine Konturen erkannt")
    
    print("Finale gefilterte Konturen:")
    if filtered_contours:
        for i, contour in enumerate(filtered_contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Approximiere Kontur für Formerkennung
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            print(f"  F{i+1}: Area={area:.0f}, Umfang={perimeter:.1f}, Eckpunkte={len(approx)}")
    else:
        print("  Keine finalen Konturen")

def visualize_depth_gradients(depth_image, grad_x, grad_y, gradient_magnitude, extent, step_name, vectors=None, output_dir=None):
    """Visualisiert Tiefenbild und seine Gradienten mit Anschlusspunkten"""
    if depth_image is None:
        print("Keine Tiefenbild-Daten für Gradient-Visualisierung")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original Tiefenbild
    cmap = plt.cm.viridis
    cmap.set_bad(color='white', alpha=0.5)
    
    im1 = axes[0,0].imshow(depth_image, extent=extent, origin='lower', cmap=cmap, interpolation='nearest')
    axes[0,0].set_title('Original Tiefenbild')
    axes[0,0].set_xlabel('X-Koordinate')
    axes[0,0].set_ylabel('Y-Koordinate')
    plt.colorbar(im1, ax=axes[0,0], label='Tiefe')
    
    # Anschlusspunkte hinzufügen
    if vectors:
        for vector in vectors:
            pos = vector['position']
            axes[0,0].plot(pos['x'], pos['y'], 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1)
            axes[0,0].text(pos['x'], pos['y'], f"  P{vector['id']}", color='white', fontweight='bold', fontsize=8)
    
    # X-Gradient
    im2 = axes[0,1].imshow(grad_x, extent=extent, origin='lower', cmap='RdBu', interpolation='nearest')
    axes[0,1].set_title('X-Gradient (∂z/∂x)')
    axes[0,1].set_xlabel('X-Koordinate')
    axes[0,1].set_ylabel('Y-Koordinate')
    plt.colorbar(im2, ax=axes[0,1], label='Gradient X')
    
    # Anschlusspunkte hinzufügen
    if vectors:
        for vector in vectors:
            pos = vector['position']
            axes[0,1].plot(pos['x'], pos['y'], 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1)
            axes[0,1].text(pos['x'], pos['y'], f"  P{vector['id']}", color='white', fontweight='bold', fontsize=8)
    
    # Y-Gradient  
    im3 = axes[1,0].imshow(grad_y, extent=extent, origin='lower', cmap='RdBu', interpolation='nearest')
    axes[1,0].set_title('Y-Gradient (∂z/∂y)')
    axes[1,0].set_xlabel('X-Koordinate')
    axes[1,0].set_ylabel('Y-Koordinate')
    plt.colorbar(im3, ax=axes[1,0], label='Gradient Y')
    
    # Anschlusspunkte hinzufügen
    if vectors:
        for vector in vectors:
            pos = vector['position']
            axes[1,0].plot(pos['x'], pos['y'], 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1)
            axes[1,0].text(pos['x'], pos['y'], f"  P{vector['id']}", color='white', fontweight='bold', fontsize=8)
    
    # Gradientenmagnitude
    im4 = axes[1,1].imshow(gradient_magnitude, extent=extent, origin='lower', cmap='hot', interpolation='nearest')
    axes[1,1].set_title('Gradientenmagnitude |∇z|')
    axes[1,1].set_xlabel('X-Koordinate')
    axes[1,1].set_ylabel('Y-Koordinate')
    plt.colorbar(im4, ax=axes[1,1], label='|Gradient|')
    
    # Anschlusspunkte hinzufügen
    if vectors:
        for vector in vectors:
            pos = vector['position']
            axes[1,1].plot(pos['x'], pos['y'], 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1)
            axes[1,1].text(pos['x'], pos['y'], f"  P{vector['id']}", color='white', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    
    # Speichere das 2x2-Layout ebenfalls als Bild
    if output_dir:
        combined_filename = os.path.join(output_dir, f"{step_name}_combined_analysis.png")
    else:
        combined_filename = f"{step_name}_combined_analysis.png"
    
    plt.savefig(combined_filename, dpi=150, bbox_inches='tight')
    print(f"Kombinierte Analyse gespeichert: {combined_filename}")
    
    # plt.show()  # Deaktiviert für Batch-Verarbeitung
    
    # Speichere auch Gradientenbilder
    plt.figure(figsize=(10, 8))
    plt.imshow(gradient_magnitude, extent=extent, origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Gradientenmagnitude')
    plt.title('Tiefenbild Gradientenmagnitude')
    plt.xlabel('X-Koordinate')
    plt.ylabel('Y-Koordinate')
    
    # Anschlusspunkte hinzufügen
    if vectors:
        for vector in vectors:
            pos = vector['position']
            plt.plot(pos['x'], pos['y'], 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1)
            plt.text(pos['x'], pos['y'], f"  P{vector['id']}", color='white', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    
    # Speichere Gradientenbilder in Ordner falls angegeben
    if output_dir:
        gradient_filename = os.path.join(output_dir, f"{step_name}_gradient_magnitude.png")
        grad_x_filename = os.path.join(output_dir, f"{step_name}_gradient_x.png")
        grad_y_filename = os.path.join(output_dir, f"{step_name}_gradient_y.png")
    else:
        gradient_filename = f"{step_name}_gradient_magnitude.png"
        grad_x_filename = f"{step_name}_gradient_x.png"
        grad_y_filename = f"{step_name}_gradient_y.png"
    
    plt.savefig(gradient_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gradientenmagnitude gespeichert: {gradient_filename}")
    
    # Speichere X-Gradient als separates Bild
    plt.figure(figsize=(10, 8))
    plt.imshow(grad_x, extent=extent, origin='lower', cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='X-Gradient (∂z/∂x)')
    plt.title('X-Gradient des Tiefenbilds')
    plt.xlabel('X-Koordinate')
    plt.ylabel('Y-Koordinate')
    
    # Anschlusspunkte hinzufügen
    if vectors:
        for vector in vectors:
            pos = vector['position']
            plt.plot(pos['x'], pos['y'], 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1)
            plt.text(pos['x'], pos['y'], f"  P{vector['id']}", color='white', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(grad_x_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"X-Gradient gespeichert: {grad_x_filename}")
    
    # Speichere Y-Gradient als separates Bild
    plt.figure(figsize=(10, 8))
    plt.imshow(grad_y, extent=extent, origin='lower', cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='Y-Gradient (∂z/∂y)')
    plt.title('Y-Gradient des Tiefenbilds')
    plt.xlabel('X-Koordinate')
    plt.ylabel('Y-Koordinate')
    
    # Anschlusspunkte hinzufügen
    if vectors:
        for vector in vectors:
            pos = vector['position']
            plt.plot(pos['x'], pos['y'], 'ro', markersize=5, markeredgecolor='white', markeredgewidth=1)
            plt.text(pos['x'], pos['y'], f"  P{vector['id']}", color='white', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(grad_y_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Y-Gradient gespeichert: {grad_y_filename}")

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

def find_step_files(data_dir="Data"):
    """Findet alle STEP-Dateien im Data-Ordner"""
    step_extensions = ['.stp', '.step', '.STP', '.STEP']
    step_files = []
    
    if not os.path.exists(data_dir):
        print(f"Warnung: Data-Ordner '{data_dir}' nicht gefunden!")
        return []
    
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path) and any(file.endswith(ext) for ext in step_extensions):
            step_files.append(file_path)
    
    return sorted(step_files)

def process_step_file(step_file):
    """Verarbeitet eine einzelne STEP-Datei"""
    print(f"\n{'='*60}")
    print(f"Verarbeite: {os.path.basename(step_file)}")
    print(f"{'='*60}")
    
    # Bestimme JSON-Dateiname basierend auf STEP-Datei
    step_name = os.path.splitext(os.path.basename(step_file))[0]
    json_file = os.path.join("Data", f"{step_name}.json")
    
    try:
        # 1. Lade STEP-Datei
        mesh = load_step_file(step_file)
        
        # 2. Lade JSON-Vektoren
        vectors = load_json_vectors(json_file)
        
        # 3. Konvertiere zu Voxeln
        voxel_grid = mesh_to_voxels(mesh, voxel_resolution=400)
        
        # 4. Erstelle Vektor-Marker
        vector_geometries = create_vector_markers(vectors)
        
        # 5. Erstelle Tiefenbild
        depth_image, extent = voxels_to_depth_image(voxel_grid, direction='z')
        
        # 6. Berechne und visualisiere Gradienten
        if depth_image is not None:
            # Erstelle Ausgabe-Ordner im Data-Verzeichnis basierend auf STEP-Dateiname
            output_dir = os.path.join("Data", step_name)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Erstelle Ausgabe-Ordner: {output_dir}")
            
            # Speichere Tiefenbild in Ordner
            output_name = os.path.join(output_dir, f"{step_name}_depth_image.png")
            save_depth_image(depth_image, output_name)
            
            # Berechne und visualisiere Gradienten
            grad_x, grad_y, gradient_magnitude = differentiate_depth_image(depth_image)
            if gradient_magnitude is not None:
                # Erkenne Konturen
                contours, binary_image, contour_image, hierarchy = detect_contours(gradient_magnitude)
                
                # Visualisiere alle Ergebnisse
                visualize_depth_gradients(depth_image, grad_x, grad_y, gradient_magnitude, extent, step_name, vectors, output_dir)
                
                # Speichere Kontur-Analyse
                if contour_image is not None:
                    save_contour_analysis(binary_image, contour_image, contours, hierarchy, extent, step_name, vectors, output_dir)
        
        print(f"[OK] Erfolgreich verarbeitet: {os.path.basename(step_file)}")
        return True
        
    except Exception as e:
        print(f"[FEHLER] Fehler bei {os.path.basename(step_file)}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Hauptfunktion"""
    import sys
    
    # Prüfe Command Line Arguments
    if len(sys.argv) > 1:
        step_file = sys.argv[1]
        # Wenn vollständiger Pfad angegeben, verwende ihn direkt
        if not os.path.dirname(step_file):
            # Wenn nur Dateiname angegeben, suche in Data-Ordner
            step_file = os.path.join("Data", step_file)
        
        print("Einzeldatei-Modus:")
        success = process_step_file(step_file)
        if success:
            print("\nVerarbeitung erfolgreich abgeschlossen.")
        else:
            print("\nVerarbeitung mit Fehlern beendet.")
    else:
        # Batch-Modus: Alle STEP-Dateien im Data-Ordner verarbeiten
        print("Batch-Modus: Suche alle STEP-Dateien im Data-Ordner...")
        step_files = find_step_files("Data")
        
        if not step_files:
            print("Keine STEP-Dateien im Data-Ordner gefunden!")
            return
        
        print(f"Gefundene STEP-Dateien: {len(step_files)}")
        for step_file in step_files:
            print(f"  - {os.path.basename(step_file)}")
        
        # Verarbeite alle Dateien
        successful = 0
        failed = 0
        
        for step_file in step_files:
            success = process_step_file(step_file)
            if success:
                successful += 1
            else:
                failed += 1
        
        # Zusammenfassung
        print(f"\n{'='*60}")
        print("BATCH-VERARBEITUNG ABGESCHLOSSEN")
        print(f"{'='*60}")
        print(f"Erfolgreich: {successful}")
        print(f"Fehlgeschlagen: {failed}")
        print(f"Gesamt: {len(step_files)}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()