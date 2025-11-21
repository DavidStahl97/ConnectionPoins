#!/usr/bin/env python3
"""
Chamber Analyzer für JSON-basiertes DataSet Format

Analysiert 3D-Mesh-Daten aus JSON und erkennt Kammer-Öffnungen
basierend auf Anschlusspunkten.

Wird von der WPF-Anwendung über Python.NET aufgerufen.
"""

import os
import sys
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate

def analyze_chambers_from_json(json_file_path, output_dir=None):
    """
    Hauptfunktion: Analysiert Kammern aus JSON-Datei

    Args:
        json_file_path: Pfad zur JSON-Datei mit PartData
        output_dir: Optionaler Ausgabe-Ordner für Visualisierungen

    Returns:
        dict: Ergebnis mit chamber_centers und Status
    """
    try:
        print(f"Analysiere Kammern aus: {json_file_path}")

        # 1. Lade JSON-Daten
        part_data = load_json_data(json_file_path)

        # 2. Erstelle Mesh aus Points und Indices
        mesh = create_mesh_from_json(part_data)

        # 3. Konvertiere zu Voxeln
        voxel_grid = mesh_to_voxels(mesh, voxel_resolution=800)

        # 4. Analysiere jeden Connection Point separat in seiner InsertDirection
        connection_points = part_data.get('ConnectionPoints', [])
        chamber_centers = []

        for cp in connection_points:
            print(f"\nAnalyzing connection point {cp['Index']}: {cp['Name']}")

            # Lese InsertDirection
            insert_dir = cp.get('InsertDirection', {'X': 0.0, 'Y': 0.0, 'Z': 1.0})
            direction_vector = np.array([insert_dir['X'], insert_dir['Y'], insert_dir['Z']])
            direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalisieren

            print(f"  InsertDirection: [{direction_vector[0]:.3f}, {direction_vector[1]:.3f}, {direction_vector[2]:.3f}]")

            # Erstelle Tiefenbild in dieser Richtung
            depth_image, extent = voxels_to_depth_image_custom_direction(voxel_grid, direction_vector)

            if depth_image is None:
                print(f"  Warning: Failed to create depth image")
                chamber_centers.append({
                    'connection_point_index': cp['Index'],
                    'connection_point_name': cp['Name'],
                    'chamber_center': None
                })
                continue

            # Erkenne Konturen
            contours, binary_image, hierarchy = detect_contours_from_depth(depth_image)

            if not contours:
                print(f"  Warning: No contours detected")
                chamber_centers.append({
                    'connection_point_index': cp['Index'],
                    'connection_point_name': cp['Name'],
                    'chamber_center': None
                })
                continue

            print(f"  Detected {len(contours)} contours")

            # Berechne Kammer-Mittelpunkt für diesen Connection Point
            chamber_center = calculate_chamber_center_for_point(
                cp,
                contours,
                depth_image,
                extent,
                direction_vector
            )

            chamber_centers.append(chamber_center)

            # Optional: Speichere Visualisierung pro Connection Point
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_visualization_for_point(
                    depth_image,
                    contours,
                    cp,
                    chamber_center,
                    extent,
                    output_dir,
                    part_data.get('PartNr', 'unknown')
                )

        # Speichere Analyse-Ergebnisse als JSON
        if output_dir:
            save_analysis_results_json(chamber_centers, output_dir, part_data.get('PartNr', 'unknown'))

        total_contours = len(chamber_centers)
        return {
            'success': True,
            'chamber_centers': chamber_centers,
            'contour_count': total_contours
        }

    except Exception as e:
        import traceback
        error_msg = f"Error analyzing chambers: {str(e)}"
        traceback.print_exc()
        return {
            'success': False,
            'error': error_msg,
            'chamber_centers': []
        }

def load_json_data(json_file_path):
    """Lädt JSON-Daten im neuen DataSet-Format"""
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validiere erforderliche Felder
    required_fields = ['PartNr', 'Graphic3d', 'ConnectionPoints']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field in JSON: {field}")

    print(f"Loaded: {data['PartNr']} - {len(data['Graphic3d']['Points'])} points, "
          f"{len(data['ConnectionPoints'])} connection points")

    return data

def create_mesh_from_json(part_data):
    """Erstellt trimesh Mesh aus JSON Points und Indices"""
    import trimesh

    graphic_data = part_data['Graphic3d']

    # Konvertiere Points zu numpy array
    points_list = graphic_data['Points']
    vertices = np.array([[p['X'], p['Y'], p['Z']] for p in points_list])

    # Konvertiere Indices zu numpy array (Dreiecke)
    indices = np.array(graphic_data['Indices'])

    # Indices müssen Tripel sein (3 Indices pro Dreieck)
    if len(indices) % 3 != 0:
        raise ValueError(f"Indices count {len(indices)} is not divisible by 3")

    faces = indices.reshape(-1, 3)

    # Erstelle trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    print(f"Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    return mesh

def mesh_to_voxels(mesh, voxel_resolution=800):
    """Konvertiert trimesh zu Open3D Voxel Grid"""
    print(f"Converting mesh to voxels (resolution: {voxel_resolution})")

    # Konvertiere zu Open3D Mesh
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.faces)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()

    # Berechne Voxel-Größe
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    max_dimension = np.max(dimensions)
    voxel_size = max_dimension / voxel_resolution

    print(f"Voxel size: {voxel_size:.6f}")

    # Konvertiere zu Voxel Grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        o3d_mesh,
        voxel_size=voxel_size
    )

    print(f"Voxel grid created: {len(voxel_grid.get_voxels())} voxels")

    return voxel_grid

def voxels_to_depth_image(voxel_grid, direction='z'):
    """Erstellt Tiefenbild aus Voxel Grid"""
    print(f"Creating depth image (direction: {direction.upper()})")

    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        print("Warning: No voxels found!")
        return None, None

    # Extrahiere Voxel-Weltkoordinaten
    voxel_coords = np.array([
        voxel_grid.get_voxel_center_coordinate(v.grid_index)
        for v in voxels
    ])

    if direction.lower() == 'z':
        # Z-Projektion (Draufsicht)
        x_coords = voxel_coords[:, 0]
        y_coords = voxel_coords[:, 1]
        depth_coords = voxel_coords[:, 2]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        voxel_size = voxel_grid.voxel_size
        resolution_x = int(np.ceil((x_max - x_min) / voxel_size)) + 1
        resolution_y = int(np.ceil((y_max - y_min) / voxel_size)) + 1

        print(f"Depth image resolution: {resolution_x} x {resolution_y}")

        # Erstelle Tiefenbild
        depth_image = np.full((resolution_y, resolution_x), np.nan)

        for i in range(len(voxel_coords)):
            x_idx = int(round((x_coords[i] - x_min) / voxel_size))
            y_idx = int(round((y_coords[i] - y_min) / voxel_size))

            if 0 <= x_idx < resolution_x and 0 <= y_idx < resolution_y:
                current_depth = depth_image[y_idx, x_idx]
                if np.isnan(current_depth) or depth_coords[i] > current_depth:
                    depth_image[y_idx, x_idx] = depth_coords[i]

        extent = [x_min, x_max, y_min, y_max]

        valid_pixels = ~np.isnan(depth_image)
        coverage = np.sum(valid_pixels) / depth_image.size * 100
        print(f"Depth image created: {depth_image.shape}, {coverage:.1f}% coverage")

        return depth_image, extent

    return None, None

def voxels_to_depth_image_custom_direction(voxel_grid, direction_vector):
    """
    Erstellt Tiefenbild aus Voxel Grid in beliebiger Richtung

    Args:
        voxel_grid: Open3D VoxelGrid
        direction_vector: Normalisierter Richtungsvektor [x, y, z]

    Returns:
        depth_image: 2D numpy array mit Tiefenwerten
        extent: [u_min, u_max, v_min, v_max] in Weltkoordinaten
    """
    print(f"Creating depth image in direction [{direction_vector[0]:.3f}, {direction_vector[1]:.3f}, {direction_vector[2]:.3f}]")

    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        print("Warning: No voxels found!")
        return None, None

    # Extrahiere Voxel-Weltkoordinaten
    voxel_coords = np.array([
        voxel_grid.get_voxel_center_coordinate(v.grid_index)
        for v in voxels
    ])

    # Erstelle Koordinatensystem basierend auf direction_vector
    # direction_vector wird zur neuen Z-Achse (Tiefenrichtung)
    z_axis = direction_vector

    # Finde zwei orthogonale Vektoren für U und V Achsen
    # Wähle einen Vektor, der nicht parallel zu z_axis ist
    if abs(z_axis[2]) < 0.9:
        up = np.array([0, 0, 1])
    else:
        up = np.array([1, 0, 0])

    u_axis = np.cross(z_axis, up)
    u_axis = u_axis / np.linalg.norm(u_axis)

    v_axis = np.cross(z_axis, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    # Projiziere Voxel-Koordinaten auf das neue Koordinatensystem
    u_coords = np.dot(voxel_coords, u_axis)
    v_coords = np.dot(voxel_coords, v_axis)
    depth_coords = np.dot(voxel_coords, z_axis)

    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()

    voxel_size = voxel_grid.voxel_size
    resolution_u = int(np.ceil((u_max - u_min) / voxel_size)) + 1
    resolution_v = int(np.ceil((v_max - v_min) / voxel_size)) + 1

    print(f"  Depth image resolution: {resolution_u} x {resolution_v}")

    # Erstelle Tiefenbild
    depth_image = np.full((resolution_v, resolution_u), np.nan)

    for i in range(len(voxel_coords)):
        u_idx = int(round((u_coords[i] - u_min) / voxel_size))
        v_idx = int(round((v_coords[i] - v_min) / voxel_size))

        if 0 <= u_idx < resolution_u and 0 <= v_idx < resolution_v:
            current_depth = depth_image[v_idx, u_idx]
            # Speichere maximale Tiefe (am weitesten in Richtung direction_vector)
            if np.isnan(current_depth) or depth_coords[i] > current_depth:
                depth_image[v_idx, u_idx] = depth_coords[i]

    extent = [u_min, u_max, v_min, v_max]

    # Speichere die Transformation für spätere Rückrechnung
    # Diese wird in extent als zusätzliche Daten gespeichert
    transform_info = {
        'u_axis': u_axis,
        'v_axis': v_axis,
        'z_axis': z_axis,
        'extent': extent,
        'voxel_size': voxel_size
    }

    valid_pixels = ~np.isnan(depth_image)
    coverage = np.sum(valid_pixels) / depth_image.size * 100
    print(f"  Depth image created: {depth_image.shape}, {coverage:.1f}% coverage")

    # Gebe extent und transform_info zurück
    return depth_image, (extent, transform_info)

def filter_nested_contours_with_hierarchy(filtered_contours, all_contours, hierarchy):
    """
    Entfernt verschachtelte Konturen basierend auf OpenCV Hierarchie-Information

    Args:
        filtered_contours: Bereits größengefilterte Konturen
        all_contours: Alle ursprünglichen Konturen (für Index-Mapping)
        hierarchy: OpenCV Hierarchie-Array [next, previous, first_child, parent]

    Returns:
        filtered_contours: Liste von Konturen ohne verschachtelte Konturen
    """
    if len(filtered_contours) <= 1 or hierarchy is None:
        return filtered_contours

    print("  Filtere verschachtelte Konturen mit OpenCV Hierarchie...")

    # Hierarchie-Format: [next, previous, first_child, parent]
    # parent == -1: Kontur ist auf oberster Ebene (nicht verschachtelt)
    # parent >= 0: Kontur hat eine Eltern-Kontur (ist verschachtelt)

    # Erstelle Mapping von Kontur-Objekten zu Original-Indizes
    contour_to_index = {}
    for i, contour in enumerate(all_contours):
        # Nutze id() als eindeutigen Identifier
        contour_to_index[id(contour)] = i

    outer_contours = []
    removed_count = 0

    for contour in filtered_contours:
        # Finde Original-Index
        contour_id = id(contour)
        if contour_id in contour_to_index:
            i = contour_to_index[contour_id]
            parent_index = hierarchy[0][i][3]  # Parent-Index

            if parent_index == -1:
                # Kontur ist auf oberster Ebene - behalten
                outer_contours.append(contour)
            else:
                # Kontur ist verschachtelt - entfernen
                print(f"    Entferne verschachtelte Kontur (Kind von C{parent_index+1})")
                removed_count += 1
        else:
            # Fallback: Kontur behalten wenn Index nicht gefunden
            outer_contours.append(contour)

    print(f"  Verschachtelte Konturen: {len(filtered_contours)} -> {len(outer_contours)} (entfernt: {removed_count})")

    return outer_contours

def detect_contours_from_depth(depth_image, threshold_factor=0.1):
    """Erkennt Konturen aus Tiefenbild mittels Gradientenanalyse"""
    print("Detecting contours from depth image")

    # Berechne Gradienten
    valid_mask = ~np.isnan(depth_image)
    depth_filled = depth_image.copy()
    depth_filled[~valid_mask] = 0

    # Sobel-Filter für Gradienten
    grad_x = cv2.Sobel(depth_filled, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_filled, cv2.CV_64F, 0, 1, ksize=3)

    # Gradientenmagnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude[~valid_mask] = 0

    # Schwellwert für Konturerkennung
    threshold = threshold_factor * np.max(gradient_magnitude)
    binary_image = (gradient_magnitude > threshold).astype(np.uint8) * 255

    # Erkenne Konturen
    contours, hierarchy = cv2.findContours(
        binary_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtere kleine Konturen
    min_contour_area = 20
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]

    # Filtere verschachtelte Konturen (innere Konturen entfernen)
    filtered_contours = filter_nested_contours_with_hierarchy(filtered_contours, contours, hierarchy)

    print(f"Detected {len(filtered_contours)} contours (filtered from {len(contours)})")

    return filtered_contours, binary_image, hierarchy

def calculate_chamber_center_for_point(cp, contours, depth_image, extent_info, direction_vector):
    """
    Berechnet Kammer-Mittelpunkt für einen einzelnen Connection Point

    Args:
        cp: Connection Point Dictionary
        contours: Liste von erkannten Konturen
        depth_image: 2D Tiefenbild
        extent_info: Tuple von (extent, transform_info)
        direction_vector: InsertDirection des Connection Points

    Returns:
        Dictionary mit chamber_center Ergebnis
    """
    extent, transform_info = extent_info

    u_min, u_max, v_min, v_max = extent
    u_axis = transform_info['u_axis']
    v_axis = transform_info['v_axis']
    z_axis = transform_info['z_axis']
    voxel_size = transform_info['voxel_size']

    # Konvertiere Connection Point Weltkoordinate zu U/V Koordinaten
    cp_point = cp['Point']
    cp_world = np.array([cp_point['X'], cp_point['Y'], cp_point['Z']])

    cp_u = np.dot(cp_world, u_axis)
    cp_v = np.dot(cp_world, v_axis)
    cp_depth = np.dot(cp_world, z_axis)

    # Konvertiere zu Pixel-Koordinaten
    pixel_u = int(round((cp_u - u_min) / voxel_size))
    pixel_v = int(round((cp_v - v_min) / voxel_size))

    print(f"  Connection point pixel: ({pixel_u}, {pixel_v})")

    # Finde enthaltende Kontur
    containing_contour = None
    for contour in contours:
        result = cv2.pointPolygonTest(contour, (pixel_u, pixel_v), False)
        if result >= 0:  # Punkt ist in oder auf Kontur
            containing_contour = contour
            break

    if containing_contour is None:
        print(f"  Warning: No contour found containing the connection point")
        return {
            'connection_point_index': cp['Index'],
            'connection_point_name': cp['Name'],
            'chamber_center': None
        }

    # Berechne Bounding Box der Kontur
    u_box, v_box, w_box, h_box = cv2.boundingRect(containing_contour)

    # Berechne Mittelpunkt in Pixelkoordinaten
    center_pixel_u = u_box + w_box / 2.0
    center_pixel_v = v_box + h_box / 2.0

    # Extrahiere Tiefe aus Tiefenbild an Kontur-Rand-Pixeln
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [containing_contour], -1, 255, thickness=1)

    contour_depths = depth_image[mask == 255]
    contour_depths = contour_depths[~np.isnan(contour_depths)]

    if len(contour_depths) > 0:
        center_depth = np.max(contour_depths)  # Maximale Tiefe
    else:
        center_depth = cp_depth  # Fallback

    # Konvertiere zurück zu U/V Weltkoordinaten
    center_u = u_min + center_pixel_u * voxel_size
    center_v = v_min + center_pixel_v * voxel_size

    # Rekonstruiere 3D-Weltkoordinate
    center_world = center_u * u_axis + center_v * v_axis + center_depth * z_axis

    print(f"  Chamber center: ({center_world[0]:.4f}, {center_world[1]:.4f}, {center_world[2]:.4f})")

    return {
        'connection_point_index': cp['Index'],
        'connection_point_name': cp['Name'],
        'chamber_center': {
            'X': float(center_world[0]),
            'Y': float(center_world[1]),
            'Z': float(center_world[2])
        }
    }

def calculate_chamber_centers(contours, connection_points, depth_image, extent, voxel_grid):
    """Alte Funktion - wird nicht mehr verwendet, für Kompatibilität behalten"""
    print("Warning: Using deprecated calculate_chamber_centers function")
    return []

def save_visualization_for_point(depth_image, contours, cp, chamber_center, extent_info, output_dir, part_nr):
    """
    Speichert erweiterte Visualisierungen für einen einzelnen Connection Point

    Args:
        depth_image: 2D Tiefenbild
        contours: Erkannte Konturen
        cp: Connection Point Dictionary
        chamber_center: Berechneter Kammer-Mittelpunkt
        extent_info: Tuple von (extent, transform_info)
        output_dir: Ausgabe-Verzeichnis
        part_nr: Teil-Nummer
    """
    extent, transform_info = extent_info
    u_min, u_max, v_min, v_max = extent
    voxel_size = transform_info['voxel_size']
    u_axis = transform_info['u_axis']
    v_axis = transform_info['v_axis']
    z_axis = transform_info['z_axis']

    # Dateiname mit Connection Point Index
    filename_base = f"{part_nr}_CP{cp['Index']}_{cp['Name'].replace(' ', '_')}"

    valid_mask = ~np.isnan(depth_image)
    depth_display = depth_image.copy()
    if np.any(valid_mask):
        depth_display[~valid_mask] = np.nanmin(depth_image)
    else:
        print(f"  Warning: No valid depth data for visualization")
        return

    # Connection Point Position in U/V
    cp_point = cp['Point']
    cp_world = np.array([cp_point['X'], cp_point['Y'], cp_point['Z']])
    cp_u = np.dot(cp_world, u_axis)
    cp_v = np.dot(cp_world, v_axis)

    # Chamber Center Position (falls vorhanden)
    cc_u, cc_v = None, None
    if chamber_center['chamber_center'] is not None:
        cc = chamber_center['chamber_center']
        cc_world = np.array([cc['X'], cc['Y'], cc['Z']])
        cc_u = np.dot(cc_world, u_axis)
        cc_v = np.dot(cc_world, v_axis)

    # 1. Tiefenbild allein
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(depth_display, extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Depth')
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_title(f'Depth Image - CP{cp["Index"]}: {cp["Name"]}')
    plt.savefig(os.path.join(output_dir, f'{filename_base}_1_depth.png'), bbox_inches='tight')
    plt.close()

    # 2. Berechne Gradienten
    depth_filled = depth_image.copy()
    depth_filled[~valid_mask] = 0

    grad_x = cv2.Sobel(depth_filled, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_filled, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude[~valid_mask] = 0

    # 3. Gradient X
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(grad_x, extent=extent, origin='lower', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Gradient X')
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_title(f'Gradient X - CP{cp["Index"]}')
    plt.savefig(os.path.join(output_dir, f'{filename_base}_2_gradient_x.png'), bbox_inches='tight')
    plt.close()

    # 4. Gradient Y
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(grad_y, extent=extent, origin='lower', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Gradient Y')
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_title(f'Gradient Y - CP{cp["Index"]}')
    plt.savefig(os.path.join(output_dir, f'{filename_base}_3_gradient_y.png'), bbox_inches='tight')
    plt.close()

    # 5. Gradient Magnitude (Länge)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(gradient_magnitude, extent=extent, origin='lower', cmap='hot')
    plt.colorbar(im, ax=ax, label='Gradient Magnitude')
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_title(f'Gradient Magnitude - CP{cp["Index"]}')
    plt.savefig(os.path.join(output_dir, f'{filename_base}_4_gradient_magnitude.png'), bbox_inches='tight')
    plt.close()

    # 6. Binary Image (nach Thresholding)
    threshold = 0.1 * np.max(gradient_magnitude)
    binary_image = (gradient_magnitude > threshold).astype(np.uint8) * 255

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.imshow(binary_image, extent=extent, origin='lower', cmap='gray')
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_title(f'Binary Image (Threshold={threshold:.2f}) - CP{cp["Index"]}')
    plt.savefig(os.path.join(output_dir, f'{filename_base}_5_binary.png'), bbox_inches='tight')
    plt.close()

    # 7. Finale Analyse mit Konturen
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(depth_display, extent=extent, origin='lower', cmap='viridis', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Depth')

    # Zeichne Konturen
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'pink']
    for i, contour in enumerate(contours):
        contour_squeezed = contour.squeeze()
        if len(contour_squeezed.shape) == 1:
            continue

        contour_u = u_min + contour_squeezed[:, 0] * voxel_size
        contour_v = v_min + contour_squeezed[:, 1] * voxel_size

        color = colors[i % len(colors)]
        ax.plot(contour_u, contour_v, color=color, linewidth=2)

    # Zeichne Connection Point (roter Punkt)
    ax.plot(cp_u, cp_v, 'ro', markersize=12,
            markeredgecolor='white', markeredgewidth=2, zorder=10)

    # Zeichne Chamber Center (grünes Dreieck)
    if cc_u is not None and cc_v is not None:
        ax.plot(cc_u, cc_v, 'g^', markersize=14,
                markeredgecolor='white', markeredgewidth=2, zorder=10)

    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_title(f'Final Analysis - CP{cp["Index"]}: {cp["Name"]}\nInsertDirection: {cp.get("InsertDirection", {})}')

    plt.savefig(os.path.join(output_dir, f'{filename_base}_6_final_analysis.png'), bbox_inches='tight')
    plt.close()

    print(f"  Saved 6 visualization images for {filename_base}")

def save_analysis_results_json(chamber_centers, output_dir, part_nr):
    """
    Speichert die Analyse-Ergebnisse als JSON-Datei

    Args:
        chamber_centers: Liste von Chamber Center Ergebnissen
        output_dir: Ausgabe-Verzeichnis
        part_nr: Teil-Nummer
    """
    output_file = os.path.join(output_dir, f'{part_nr}_chamber_analysis_results.json')

    results = {
        'part_nr': part_nr,
        'analysis_timestamp': __import__('datetime').datetime.now().isoformat(),
        'chamber_centers': chamber_centers
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Saved analysis results to: {output_file}")

def save_visualizations(depth_image, contours, connection_points, chamber_centers, extent, output_dir, part_nr):
    """Alte Funktion - wird nicht mehr verwendet"""
    print(f"Saving visualizations to: {output_dir}")

    # 1. Tiefenbild
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    valid_mask = ~np.isnan(depth_image)
    depth_display = depth_image.copy()
    depth_display[~valid_mask] = np.nanmin(depth_image)

    im = ax.imshow(depth_display, extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Depth (Z)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Depth Image - {part_nr}')

    plt.savefig(os.path.join(output_dir, f'{part_nr}_depth_image.png'), bbox_inches='tight')
    plt.close()

    # 2. Konturen mit Connection Points
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Zeichne Tiefenbild als Hintergrund
    ax.imshow(depth_display, extent=extent, origin='lower', cmap='gray', alpha=0.3)

    # Zeichne Konturen
    voxel_size = (extent[1] - extent[0]) / depth_image.shape[1]
    for i, contour in enumerate(contours):
        contour_world = contour.squeeze()
        if len(contour_world.shape) == 1:
            continue

        contour_x = extent[0] + contour_world[:, 0] * voxel_size
        contour_y = extent[2] + contour_world[:, 1] * voxel_size

        ax.plot(contour_x, contour_y, linewidth=2, label=f'Contour {i+1}')

    # Zeichne Connection Points
    for cp in connection_points:
        ax.plot(cp['Point']['X'], cp['Point']['Y'], 'ro', markersize=8, label='Connection Point')

    # Zeichne Chamber Centers
    for cc in chamber_centers:
        if cc['chamber_center'] is not None:
            ax.plot(cc['chamber_center']['X'], cc['chamber_center']['Y'], 'g^',
                   markersize=10, label='Chamber Center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Contours and Chamber Centers - {part_nr}')
    ax.legend()

    plt.savefig(os.path.join(output_dir, f'{part_nr}_contours_analysis.png'), bbox_inches='tight')
    plt.close()

    print("Visualizations saved")

def main():
    """Hauptfunktion für Command Line Ausführung"""
    if len(sys.argv) < 2:
        print("Usage: python chamber_analyzer_json.py <json_file_path> [output_dir]")
        sys.exit(1)

    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_chambers_from_json(json_file, output_dir)

    if result['success']:
        print(f"\nAnalysis successful!")
        print(f"Chamber centers found: {len(result['chamber_centers'])}")

        # Ausgabe der Ergebnisse
        for cc in result['chamber_centers']:
            print(f"  {cc['connection_point_name']}: {cc['chamber_center']}")
    else:
        print(f"\nAnalysis failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
