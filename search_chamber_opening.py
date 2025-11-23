#!/usr/bin/env python3
"""
Kammeröffnungs-Suche mit schichtbasiertem Ansatz

Dieses Skript implementiert ein neues Verfahren zur Erkennung von Kammeröffnungen:
1. Voxelisierung des 3D-Objekts entlang der Anschlussvektor-Achse
2. Berechnung des 2D Boundary (wie beim alten Verfahren mit dilate)
3. Extraktion aller 2D-Schichten (Stufen) von Stufe 0 (Anschlusspunkt) bis Max Stufe
4. Später: Binary Search über die Schichten zur Öffnungserkennung
"""

import os
import json
import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation


def load_json_data(json_file_path):
    """Lädt JSON-Datei im DataSet-Format mit 3D-Mesh-Daten und ConnectionPoints"""
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {json_file_path}")

    print(f"Lade JSON-Datei: {json_file_path}")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validiere erforderliche Felder
    if 'Graphic3d' not in data:
        raise ValueError("JSON-Datei enthält kein 'Graphic3d' Feld!")
    if 'ConnectionPoints' not in data:
        raise ValueError("JSON-Datei enthält kein 'ConnectionPoints' Feld!")

    # Lade Mesh-Daten aus Graphic3d
    graphic3d = data['Graphic3d']

    if 'Points' not in graphic3d or 'Indices' not in graphic3d:
        raise ValueError("Graphic3d muss 'Points' und 'Indices' enthalten!")

    # Konvertiere Points zu Vertices (X, Y, Z -> array)
    points = graphic3d['Points']
    vertices = np.array([[p['X'], p['Y'], p['Z']] for p in points])

    # Konvertiere Indices zu Faces (flache Liste -> Triplets)
    indices = graphic3d['Indices']
    if len(indices) % 3 != 0:
        raise ValueError(f"Indices-Anzahl ({len(indices)}) ist nicht durch 3 teilbar!")

    faces = np.array(indices).reshape(-1, 3)

    # Erstelle Mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(f"Mesh geladen: {len(vertices)} Vertices, {len(faces)} Faces")

    # Lade ConnectionPoints und konvertiere zu einheitlichem Format
    connection_points = data['ConnectionPoints']
    vectors = []

    for cp in connection_points:
        vector = {
            'id': cp['Index'],
            'name': cp.get('Name', f"Point_{cp['Index']}"),
            'position': {
                'x': cp['Point']['X'],
                'y': cp['Point']['Y'],
                'z': cp['Point']['Z']
            },
            'direction': {
                'x': cp['InsertDirection']['X'],
                'y': cp['InsertDirection']['Y'],
                'z': cp['InsertDirection']['Z']
            }
        }
        vectors.append(vector)

    print(f"ConnectionPoints geladen: {len(vectors)} Anschlusspunkte")

    return mesh, vectors


def rotate_mesh_to_align_vector(mesh, direction_vector, target_direction=np.array([0, 0, 1])):
    """
    Rotiert das Mesh so, dass der Anschlussvektor in Richtung target_direction zeigt

    Args:
        mesh: trimesh Mesh-Objekt
        direction_vector: Richtungsvektor der in Z-Richtung zeigen soll (als dict oder array)
        target_direction: Zielrichtung (default: [0, 0, 1])

    Returns:
        rotated_mesh: Rotiertes Mesh
        rotation_matrix: Die verwendete Rotationsmatrix
    """
    # Konvertiere direction_vector zu numpy array falls nötig
    if isinstance(direction_vector, dict):
        dir_vec = np.array([direction_vector['x'], direction_vector['y'], direction_vector['z']])
    else:
        dir_vec = np.array(direction_vector)

    # Normalisiere Vektoren
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    target_direction = target_direction / np.linalg.norm(target_direction)

    # Berechne Rotationsachse (Kreuzprodukt)
    rotation_axis = np.cross(dir_vec, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    # Prüfe ob Vektoren bereits parallel sind
    if rotation_axis_norm < 1e-6:
        # Vektoren sind bereits parallel
        if np.dot(dir_vec, target_direction) > 0:
            # Gleiche Richtung - keine Rotation nötig
            print("Anschlussvektor zeigt bereits in Z-Richtung, keine Rotation nötig")
            return mesh.copy(), np.eye(4)
        else:
            # Entgegengesetzte Richtung - 180° Rotation um beliebige orthogonale Achse
            rotation_axis = np.array([1, 0, 0]) if abs(dir_vec[0]) < 0.9 else np.array([0, 1, 0])
            rotation_angle = np.pi
    else:
        # Normalisiere Rotationsachse
        rotation_axis = rotation_axis / rotation_axis_norm

        # Berechne Rotationswinkel
        rotation_angle = np.arccos(np.clip(np.dot(dir_vec, target_direction), -1.0, 1.0))

    # Erstelle Rotationsmatrix mit scipy
    rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
    rotation_matrix_3x3 = rotation.as_matrix()

    # Konvertiere zu 4x4 Transformationsmatrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation_matrix_3x3

    # Rotiere das Mesh
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix)

    print(f"Mesh rotiert: Anschlussvektor {dir_vec} -> {target_direction}")
    print(f"Rotationswinkel: {np.degrees(rotation_angle):.2f}°")

    return rotated_mesh, rotation_matrix


def mesh_to_voxels(mesh, voxel_resolution=800):
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

    # Berechne Voxel-Größe
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    max_dimension = np.max(dimensions)
    voxel_size = max_dimension / voxel_resolution

    print(f"Berechnete Voxel-Größe: {voxel_size:.6f}")

    # Konvertiere zu Voxel Grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        o3d_mesh,
        voxel_size=voxel_size
    )

    print(f"Voxel Grid erstellt: {len(voxel_grid.get_voxels())} Voxel")
    return voxel_grid, voxel_size


def extract_voxel_grid_as_3d_array(voxel_grid):
    """
    Extrahiert Voxel Grid als 3D numpy array

    Returns:
        voxel_array: 3D boolean array (True = Voxel vorhanden)
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max) in Weltkoordinaten
        grid_shape: (nx, ny, nz) Anzahl der Voxel in jeder Dimension
    """
    print("Extrahiere Voxel Grid als 3D Array...")

    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        print("Warnung: Keine Voxel gefunden!")
        return None, None, None

    # Extrahiere alle Voxel-Koordinaten
    voxel_coords = []
    for voxel in voxels:
        world_coord = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        grid_idx = voxel.grid_index
        voxel_coords.append((*world_coord, *grid_idx))

    voxel_coords = np.array(voxel_coords)

    # Weltkoordinaten
    world_coords = voxel_coords[:, :3]
    x_min, y_min, z_min = world_coords.min(axis=0)
    x_max, y_max, z_max = world_coords.max(axis=0)

    # Grid-Indizes
    grid_indices = voxel_coords[:, 3:].astype(int)
    ix_min, iy_min, iz_min = grid_indices.min(axis=0)
    ix_max, iy_max, iz_max = grid_indices.max(axis=0)

    # Erstelle 3D Array
    grid_shape = (ix_max - ix_min + 1, iy_max - iy_min + 1, iz_max - iz_min + 1)
    voxel_array = np.zeros(grid_shape, dtype=bool)

    # Fülle Array mit Voxeln
    for coord in grid_indices:
        ix, iy, iz = coord - np.array([ix_min, iy_min, iz_min])
        voxel_array[ix, iy, iz] = True

    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

    print(f"3D Array erstellt: {grid_shape} (x={grid_shape[0]}, y={grid_shape[1]}, z={grid_shape[2]})")
    print(f"Bounds: X=[{x_min:.4f}, {x_max:.4f}], Y=[{y_min:.4f}, {y_max:.4f}], Z=[{z_min:.4f}, {z_max:.4f}]")
    print(f"Gefüllte Voxel: {voxel_array.sum()} / {voxel_array.size} ({100*voxel_array.sum()/voxel_array.size:.2f}%)")

    return voxel_array, bounds, grid_shape


def create_2d_boundary(voxel_array, dilate_iterations=2, kernel_size=3):
    """
    Erstellt 2D Boundary durch Projektion und morphologische Operationen

    Args:
        voxel_array: 3D boolean array
        dilate_iterations: Anzahl der Dilate-Iterationen
        kernel_size: Größe des morphologischen Kernels

    Returns:
        boundary_image: 2D binary image mit Boundary
        projection_image: 2D Projektion (max-Projektion in Z)
    """
    print("Erstelle 2D Boundary...")

    # Max-Projektion in Z-Richtung (oder any-Projektion für binär)
    projection = np.any(voxel_array, axis=2)  # Projiziere entlang Z

    # Konvertiere zu uint8 für OpenCV
    projection_image = (projection * 255).astype(np.uint8)

    print(f"Projektion erstellt: {projection_image.shape}, {projection.sum()} gefüllte Pixel")

    # Morphologische Operationen für Boundary-Erkennung
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate um Objekt zu erweitern
    dilated = cv2.dilate(projection_image, kernel, iterations=dilate_iterations)

    # Gradientenberechnung für Boundary
    grad_x = cv2.Sobel(dilated, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dilated, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Schwellenwert für Boundary
    threshold = gradient_magnitude.max() * 0.3
    boundary_image = (gradient_magnitude > threshold).astype(np.uint8) * 255

    print(f"Boundary erstellt: {(boundary_image > 0).sum()} Boundary-Pixel")

    return boundary_image, projection_image


def find_connection_point_z_level(connection_point, bounds, grid_shape, voxel_size):
    """
    Findet die Z-Stufe (Layer-Index) des Anschlusspunkts

    Args:
        connection_point: Position des Anschlusspunkts (dict mit x, y, z)
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        grid_shape: (nx, ny, nz)
        voxel_size: Größe eines Voxels

    Returns:
        z_level: Index der Z-Stufe (0-basiert)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Konvertiere Weltkoordinate zu Grid-Index
    z_world = connection_point['z']
    z_normalized = (z_world - z_min) / (z_max - z_min)
    z_level = int(z_normalized * (grid_shape[2] - 1))

    # Clamp zu gültigem Bereich
    z_level = max(0, min(grid_shape[2] - 1, z_level))

    print(f"Anschlusspunkt Z-Koordinate: {z_world:.6f}")
    print(f"Z-Stufe (Layer-Index): {z_level} / {grid_shape[2] - 1}")

    return z_level


def extract_2d_layer(voxel_array, z_level):
    """
    Extrahiert eine 2D-Schicht (Stufe) aus dem 3D Voxel Array

    Args:
        voxel_array: 3D boolean array
        z_level: Z-Stufe Index

    Returns:
        layer_image: 2D uint8 image (0 oder 255)
    """
    if z_level < 0 or z_level >= voxel_array.shape[2]:
        print(f"Warnung: Z-Level {z_level} außerhalb des gültigen Bereichs [0, {voxel_array.shape[2] - 1}]")
        return None

    # Extrahiere 2D Slice bei z_level
    layer = voxel_array[:, :, z_level]

    # Konvertiere zu uint8 Bild
    layer_image = (layer * 255).astype(np.uint8)

    return layer_image


def save_all_layer_images(voxel_array, output_dir, step_name, z_start=0, z_end=None):
    """
    Speichert alle 2D-Schichten als Bilder

    Args:
        voxel_array: 3D boolean array
        output_dir: Ausgabe-Verzeichnis
        step_name: Name für Dateibenennungen
        z_start: Start Z-Level (Stufe 0 = Anschlusspunkt)
        z_end: End Z-Level (None = bis zum Ende)

    Returns:
        saved_files: Liste der gespeicherten Dateipfade
    """
    print(f"Speichere alle 2D-Schichten von Stufe {z_start} bis {z_end or 'Ende'}...")

    if z_end is None:
        z_end = voxel_array.shape[2] - 1

    # Erstelle Unterordner für Layer-Bilder
    layers_dir = os.path.join(output_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)

    saved_files = []

    for z_level in range(z_start, z_end + 1):
        layer_image = extract_2d_layer(voxel_array, z_level)

        if layer_image is not None:
            # Dateiname mit führenden Nullen für korrekte Sortierung
            filename = os.path.join(layers_dir, f"{step_name}_layer_{z_level:04d}.png")
            cv2.imwrite(filename, layer_image)
            saved_files.append(filename)

    print(f"Gespeichert: {len(saved_files)} Layer-Bilder in {layers_dir}")

    return saved_files


def visualize_layers_overview(voxel_array, output_dir, step_name, z_start, z_end, sample_count=16):
    """
    Erstellt Übersichts-Visualisierung mit mehreren Layer-Samples

    Args:
        voxel_array: 3D boolean array
        output_dir: Ausgabe-Verzeichnis
        step_name: Name für Dateibenennungen
        z_start: Start Z-Level
        z_end: End Z-Level
        sample_count: Anzahl der zu zeigenden Samples
    """
    print(f"Erstelle Layer-Übersicht mit {sample_count} Samples...")

    # Wähle gleichmäßig verteilte Samples
    if z_end - z_start + 1 < sample_count:
        sample_count = z_end - z_start + 1

    sample_indices = np.linspace(z_start, z_end, sample_count, dtype=int)

    # Berechne Grid-Layout
    cols = 4
    rows = (sample_count + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if sample_count > 1 else [axes]

    for idx, z_level in enumerate(sample_indices):
        layer_image = extract_2d_layer(voxel_array, z_level)

        axes[idx].imshow(layer_image, cmap='gray', origin='lower')
        axes[idx].set_title(f'Stufe {z_level}')
        axes[idx].axis('off')

    # Deaktiviere übrige Subplots
    for idx in range(len(sample_indices), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    overview_filename = os.path.join(output_dir, f"{step_name}_layers_overview.png")
    plt.savefig(overview_filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Layer-Übersicht gespeichert: {overview_filename}")


def process_connection_point(json_file, connection_point_id=None):
    """
    Verarbeitet einen Anschlusspunkt und erstellt alle Layer-Bilder

    Args:
        json_file: Pfad zur JSON-Datei mit Anschlusspunkten
        connection_point_id: ID des zu verarbeitenden Anschlusspunkts (None = erster)
    """
    print(f"\n{'='*60}")
    print(f"Verarbeite: {os.path.basename(json_file)}")
    print(f"{'='*60}")

    try:
        # 1. Lade JSON-Datei (enthält Mesh-Daten und Anschlussvektoren)
        mesh, vectors = load_json_data(json_file)

        if not vectors:
            raise ValueError("Keine Anschlussvektoren in JSON-Datei gefunden!")

        # Bestimme Ausgabe-Name aus JSON-Datei
        json_name = os.path.splitext(os.path.basename(json_file))[0]

        # 2. Wähle Anschlusspunkt
        if connection_point_id is not None:
            vector = next((v for v in vectors if v['id'] == connection_point_id), None)
            if vector is None:
                print(f"Anschlusspunkt mit ID {connection_point_id} nicht gefunden!")
                return False
        else:
            vector = vectors[0]  # Verwende ersten Anschlusspunkt

        print(f"\nVerarbeite Anschlusspunkt P{vector['id']}")
        print(f"  Position: ({vector['position']['x']:.6f}, {vector['position']['y']:.6f}, {vector['position']['z']:.6f})")
        print(f"  Richtung: ({vector['direction']['x']:.6f}, {vector['direction']['y']:.6f}, {vector['direction']['z']:.6f})")

        # 3. Rotiere Mesh so dass Anschlussvektor in Z-Richtung zeigt
        rotated_mesh, rotation_matrix = rotate_mesh_to_align_vector(mesh, vector['direction'])

        # Rotiere auch den Anschlusspunkt
        connection_point_world = np.array([
            vector['position']['x'],
            vector['position']['y'],
            vector['position']['z'],
            1.0
        ])
        rotated_connection_point = rotation_matrix @ connection_point_world
        rotated_connection_dict = {
            'x': rotated_connection_point[0],
            'y': rotated_connection_point[1],
            'z': rotated_connection_point[2]
        }

        print(f"  Rotierter Anschlusspunkt: ({rotated_connection_dict['x']:.6f}, {rotated_connection_dict['y']:.6f}, {rotated_connection_dict['z']:.6f})")

        # 4. Konvertiere zu Voxeln
        voxel_grid, voxel_size = mesh_to_voxels(rotated_mesh, voxel_resolution=800)

        # 5. Extrahiere als 3D Array
        voxel_array, bounds, grid_shape = extract_voxel_grid_as_3d_array(voxel_grid)

        if voxel_array is None:
            raise ValueError("Fehler beim Erstellen des Voxel Arrays!")

        # 6. Erstelle Ausgabe-Ordner
        output_dir = os.path.join("Data", f"{json_name}_P{vector['id']}_search")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nErstelle Ausgabe-Ordner: {output_dir}")

        # 7. Erstelle und speichere 2D Boundary
        boundary_image, projection_image = create_2d_boundary(voxel_array)

        boundary_filename = os.path.join(output_dir, f"{json_name}_P{vector['id']}_boundary.png")
        cv2.imwrite(boundary_filename, boundary_image)
        print(f"2D Boundary gespeichert: {boundary_filename}")

        projection_filename = os.path.join(output_dir, f"{json_name}_P{vector['id']}_projection.png")
        cv2.imwrite(projection_filename, projection_image)
        print(f"2D Projektion gespeichert: {projection_filename}")

        # 8. Finde Z-Stufe des Anschlusspunkts
        z_start = find_connection_point_z_level(rotated_connection_dict, bounds, grid_shape, voxel_size)
        z_end = grid_shape[2] - 1

        print(f"\nZ-Bereich: Stufe {z_start} (Anschlusspunkt) bis Stufe {z_end} (Max)")
        print(f"Anzahl Stufen: {z_end - z_start + 1}")

        # 9. Speichere alle Layer-Bilder
        saved_files = save_all_layer_images(voxel_array, output_dir, f"{json_name}_P{vector['id']}", z_start, z_end)

        # 10. Erstelle Übersichts-Visualisierung
        visualize_layers_overview(voxel_array, output_dir, f"{json_name}_P{vector['id']}", z_start, z_end, sample_count=16)

        print(f"\n[OK] Erfolgreich verarbeitet: {len(saved_files)} Layer-Bilder erstellt")
        return True

    except Exception as e:
        print(f"[FEHLER] Fehler bei Verarbeitung: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Hauptfunktion"""
    import sys

    # Prüfe Command Line Arguments
    if len(sys.argv) > 1:
        json_file = sys.argv[1]

        # Wenn nur Dateiname angegeben, suche in Data-Ordner
        if not os.path.dirname(json_file):
            json_file = os.path.join("Data", json_file)

        # Optional: Anschlusspunkt-ID als zweites Argument
        connection_point_id = None
        if len(sys.argv) > 2:
            connection_point_id = int(sys.argv[2])

        print("Verarbeite JSON-Datei:")
        success = process_connection_point(json_file, connection_point_id)

        if success:
            print("\nVerarbeitung erfolgreich abgeschlossen.")
        else:
            print("\nVerarbeitung mit Fehlern beendet.")
    else:
        print("Verwendung:")
        print("  python search_chamber_opening.py <json_file> [connection_point_id]")
        print("\nBeispiele:")
        print("  python search_chamber_opening.py Data/model.json")
        print("  python search_chamber_opening.py model.json 1")
        print("\nHinweis:")
        print("  - Die JSON-Datei muss Mesh-Daten (vertices/faces) und connection_vectors enthalten")
        print("  - Anschlusspunkt-ID ist optional (Standard: erster Punkt)")


if __name__ == "__main__":
    main()
