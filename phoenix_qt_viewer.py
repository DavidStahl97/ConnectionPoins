#!/usr/bin/env python3
"""
PyQt6 3D Viewer für Phoenix Contact Terminal mit Punktauswahl
"""

import sys
import json
import os
import numpy as np
import trimesh
import open3d as o3d
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QLabel, 
                             QListWidget, QMessageBox, QStatusBar, QFileDialog, QCheckBox)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QAction
from OpenGL.GL import *
from OpenGL.arrays import vbo
import math

class Phoenix3DViewer(QOpenGLWidget):
    """OpenGL 3D-Viewer für Phoenix Contact Terminal"""
    
    pointSelected = pyqtSignal(float, float, float)
    
    def __init__(self):
        super().__init__()
        self.mesh_vertices = None
        self.mesh_faces = None
        self.selected_vectors = []
        self.vector_markers = []
        self.show_mesh = True
        self.show_coordinate_system = True  # Zeige Weltkoordinatensystem
        self.original_vertices = None  # Backup der ursprünglichen Vertices
        
        # Kamera-Parameter
        self.camera_distance = 0.1
        self.camera_rotation_x = -20
        self.camera_rotation_y = 45
        self.last_mouse_pos = None
        
        # Tastatur-Focus für Zoom
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Mesh-Daten
        self.vertex_buffer = None
        self.loaded = False
        
        # OpenGL Performance-Optimierungen
        self.display_list = None
        self.vertex_array = None
        self.normal_array = None
        self.face_array = None
        
    def load_mesh_data(self, vertices, faces):
        """Lädt Mesh-Daten in den Viewer"""
        self.original_vertices = np.array(vertices, dtype=np.float32)  # Backup der ursprünglichen Vertices
        self.mesh_vertices = np.array(vertices, dtype=np.float32)
        self.mesh_faces = np.array(faces, dtype=np.uint32)
        
        # Berechne optimale Kamera-Distanz für vollständige Sichtbarkeit
        self.fit_object_to_view()
        
        # Pre-calculate normals für bessere Performance
        self.calculate_normals()
        
        self.loaded = True
        self.update()
        
    def fit_object_to_view(self):
        """Skaliert und positioniert das Objekt optimal im Viewport"""
        if self.mesh_vertices is None or len(self.mesh_vertices) == 0:
            return
            
        # Berechne Bounding Box
        min_coords = np.min(self.mesh_vertices, axis=0)
        max_coords = np.max(self.mesh_vertices, axis=0)
        
        # Berechne Zentrum und Dimensionen
        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords
        max_dimension = np.max(dimensions)
        
        print(f"Objekt Bounding Box: Min={min_coords}, Max={max_coords}")
        print(f"Zentrum: {center}, Max Dimension: {max_dimension}")
        
        # Normalisiere das Mesh auf eine einheitliche Größe
        if max_dimension > 0:
            # Skaliere auf Größe 1.0
            self.mesh_vertices = (self.mesh_vertices - center) / max_dimension
            # Speichere Transformationsparameter für Ray-Casting
            self.mesh_center = center
            self.mesh_scale = max_dimension
            
        # Setze optimale Kamera-Distanz
        # Basierend auf Frustum-Einstellungen (0.1 bis 10.0)
        # Für ein normalisiertes Objekt der Größe 1.0
        self.camera_distance = 2.5  # Genug Abstand um das ganze Objekt zu sehen
        
        # Optimale Start-Rotation für gute Übersicht
        self.camera_rotation_x = -20
        self.camera_rotation_y = 45
        
        print(f"Kamera auf optimale Position gesetzt: Distanz={self.camera_distance}")
        
    def calculate_normals(self):
        """Berechnet alle Normalen vorab für bessere Performance"""
        num_faces = len(self.mesh_faces)
        self.face_normals = np.zeros((num_faces, 3), dtype=np.float32)
        
        for i, face in enumerate(self.mesh_faces):
            v0, v1, v2 = self.mesh_vertices[face]
            
            # Berechne Face-Normale
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            
            # Normalisiere
            length = np.linalg.norm(normal)
            if length > 0:
                normal = normal / length
                
            self.face_normals[i] = normal
            
        print(f"Normalen für {num_faces} Faces berechnet")
        
    def initializeGL(self):
        """Initialisiert OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Performance-Optimierungen
        glEnable(GL_CULL_FACE)  # Back-face culling
        glCullFace(GL_BACK)
        
        # Vertex Arrays aktivieren
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        
        # Lichteinstellungen
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Material-Eigenschaften
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.7, 0.7, 0.7, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        glClearColor(0.9, 0.9, 0.9, 1.0)  # Hellgrauer Hintergrund
        
    def resizeGL(self, width, height):
        """Behandelt Größenänderungen"""
        if height == 0:
            height = 1
            
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height
        glFrustum(-0.1 * aspect, 0.1 * aspect, -0.1, 0.1, 0.1, 10.0)
        
    def paintGL(self):
        """Zeichnet die 3D-Szene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if not self.loaded:
            return
            
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Kamera-Position
        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)
        
        # Zeichne Mesh nur wenn aktiviert
        if self.show_mesh:
            self.draw_mesh_optimized()
        
        # Zeichne ausgewählte Vektoren als Pfeile
        self.draw_vectors()
        
        # Zeichne Weltkoordinatensystem
        if self.show_coordinate_system:
            self.draw_world_coordinate_system()
        
    def create_display_list(self):
        """Erstellt eine Display List für optimiertes Rendering"""
        if self.mesh_vertices is None or self.mesh_faces is None:
            return
            
        print("Erstelle Display List für optimiertes Rendering...")
        
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        glEnable(GL_LIGHTING)
        glColor3f(0.7, 0.7, 0.7)  # Phoenix Contact Grau
        
        # Verwende Vertex Arrays für bessere Performance
        flat_vertices = []
        flat_normals = []
        
        for i, face in enumerate(self.mesh_faces):
            v0, v1, v2 = self.mesh_vertices[face]
            normal = self.face_normals[i]
            
            # Vertices
            flat_vertices.extend([v0[0], v0[1], v0[2]])
            flat_vertices.extend([v1[0], v1[1], v1[2]])
            flat_vertices.extend([v2[0], v2[1], v2[2]])
            
            # Normalen (gleiche für alle 3 Vertices des Faces)
            flat_normals.extend([normal[0], normal[1], normal[2]])
            flat_normals.extend([normal[0], normal[1], normal[2]])
            flat_normals.extend([normal[0], normal[1], normal[2]])
        
        # Konvertiere zu NumPy Arrays
        vertex_array = np.array(flat_vertices, dtype=np.float32)
        normal_array = np.array(flat_normals, dtype=np.float32)
        
        # Setze Vertex und Normal Arrays
        glVertexPointer(3, GL_FLOAT, 0, vertex_array)
        glNormalPointer(GL_FLOAT, 0, normal_array)
        
        # Zeichne alle Triangles auf einmal
        glDrawArrays(GL_TRIANGLES, 0, len(self.mesh_faces) * 3)
        
        glEndList()
        print(f"Display List erstellt mit {len(self.mesh_faces)} Faces")
        
    def draw_mesh_optimized(self):
        """Zeichnet das Mesh mit optimierter Display List"""
        if self.mesh_vertices is None or self.mesh_faces is None:
            return
            
        # Erstelle Display List beim ersten Aufruf
        if self.display_list is None:
            self.create_display_list()
            
        # Verwende Display List für optimiertes Rendering
        if self.display_list is not None:
            glCallList(self.display_list)
        
    def draw_vectors(self):
        """Zeichnet die ausgewählten Vektoren als Pfeile"""
        if not self.selected_vectors:
            return
            
        glDisable(GL_LIGHTING)
        
        for i, vector_data in enumerate(self.selected_vectors):
            start_point = vector_data['start_point']
            direction = vector_data['direction']
            
            # Transformiere Punkt in normalisierte Koordinaten für das Rendering
            if hasattr(self, 'mesh_center') and hasattr(self, 'mesh_scale'):
                normalized_start = [(start_point[j] - self.mesh_center[j]) / self.mesh_scale for j in range(3)]
                arrow_length = 0.1  # Pfeillänge in normalisierten Koordinaten
            else:
                normalized_start = start_point
                arrow_length = 0.1
            
            # Berechne Endpunkt des Pfeils
            normalized_end = [
                normalized_start[0] + direction[0] * arrow_length,
                normalized_start[1] + direction[1] * arrow_length, 
                normalized_start[2] + direction[2] * arrow_length
            ]
            
            # Zeichne Pfeil
            self.draw_arrow(normalized_start, normalized_end, direction)
        
        glEnable(GL_LIGHTING)
        
    def draw_cube(self, center, size):
        """Zeichnet einen Würfel an der gegebenen Position"""
        half_size = size / 2
        x, y, z = center
        
        glColor3f(1.0, 0.0, 0.0)  # Rot
        
        # Würfel mit 6 Flächen
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(x - half_size, y - half_size, z + half_size)
        glVertex3f(x + half_size, y - half_size, z + half_size)
        glVertex3f(x + half_size, y + half_size, z + half_size)
        glVertex3f(x - half_size, y + half_size, z + half_size)
        
        # Back face
        glVertex3f(x - half_size, y - half_size, z - half_size)
        glVertex3f(x - half_size, y + half_size, z - half_size)
        glVertex3f(x + half_size, y + half_size, z - half_size)
        glVertex3f(x + half_size, y - half_size, z - half_size)
        
        # Top face
        glVertex3f(x - half_size, y + half_size, z - half_size)
        glVertex3f(x - half_size, y + half_size, z + half_size)
        glVertex3f(x + half_size, y + half_size, z + half_size)
        glVertex3f(x + half_size, y + half_size, z - half_size)
        
        # Bottom face
        glVertex3f(x - half_size, y - half_size, z - half_size)
        glVertex3f(x + half_size, y - half_size, z - half_size)
        glVertex3f(x + half_size, y - half_size, z + half_size)
        glVertex3f(x - half_size, y - half_size, z + half_size)
        
        # Right face
        glVertex3f(x + half_size, y - half_size, z - half_size)
        glVertex3f(x + half_size, y + half_size, z - half_size)
        glVertex3f(x + half_size, y + half_size, z + half_size)
        glVertex3f(x + half_size, y - half_size, z + half_size)
        
        # Left face
        glVertex3f(x - half_size, y - half_size, z - half_size)
        glVertex3f(x - half_size, y - half_size, z + half_size)
        glVertex3f(x - half_size, y + half_size, z + half_size)
        glVertex3f(x - half_size, y + half_size, z - half_size)
        
        glEnd()
        
    def draw_world_coordinate_system(self):
        """Zeichnet das Weltkoordinatensystem mit X,Y,Z-Achsen"""
        glDisable(GL_LIGHTING)
        glLineWidth(4.0)
        
        # Koordinatenursprung (0,0,0) 
        origin = [0.0, 0.0, 0.0]
        axis_length = 0.3  # Achsenlänge in normalisierten Koordinaten
        
        # X-Achse (Rot)
        glColor3f(1.0, 0.0, 0.0)  
        glBegin(GL_LINES)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0] + axis_length, origin[1], origin[2])
        glEnd()
        
        # Y-Achse (Grün)
        glColor3f(0.0, 1.0, 0.0)  
        glBegin(GL_LINES)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0], origin[1] + axis_length, origin[2])
        glEnd()
        
        # Z-Achse (Blau)
        glColor3f(0.0, 0.0, 1.0)  
        glBegin(GL_LINES)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0], origin[1], origin[2] + axis_length)
        glEnd()
        
        # Kleine Kugel am Ursprung (Weiß)
        glColor3f(1.0, 1.0, 1.0)
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glVertex3f(origin[0], origin[1], origin[2])
        glEnd()
        glPointSize(1.0)
        
        glLineWidth(1.0)  # Zurück zur Standard-Linienbreite
        glEnable(GL_LIGHTING)
        
    def draw_arrow(self, start_point, end_point, direction):
        """Zeichnet einen kompletten Pfeil von start_point zu end_point"""
        glColor3f(1.0, 0.0, 0.0)  # Rot
        glLineWidth(3.0)
        
        # Zeichne Pfeilschaft (Linie vom Start- zum Endpunkt)
        glBegin(GL_LINES)
        glVertex3f(start_point[0], start_point[1], start_point[2])
        glVertex3f(end_point[0], end_point[1], end_point[2])
        glEnd()
        
        # Zeichne Pfeilspitze
        self.draw_arrow_head(start_point, end_point, direction)
        
        glLineWidth(1.0)  # Zurück zur Standard-Linienbreite
        
    def draw_arrow_head(self, start_point, end_point, direction):
        """Zeichnet eine einfache Pfeilspitze"""
        import math
        
        # Pfeilspitze Parameter
        head_length = 0.03  # Länge der Pfeilspitze
        head_width = 0.015   # Breite der Pfeilspitze
        
        # Berechne einen senkrechten Vektor zu direction
        # Wähle eine Achse die am wenigsten parallel zu direction ist
        if abs(direction[0]) < abs(direction[1]) and abs(direction[0]) < abs(direction[2]):
            up = [1.0, 0.0, 0.0]
        elif abs(direction[1]) < abs(direction[2]):
            up = [0.0, 1.0, 0.0]
        else:
            up = [0.0, 0.0, 1.0]
        
        # Berechne senkrechten Vektor mit Kreuzprodukt
        perpendicular1 = [
            direction[1] * up[2] - direction[2] * up[1],
            direction[2] * up[0] - direction[0] * up[2],
            direction[0] * up[1] - direction[1] * up[0]
        ]
        
        # Normalisiere perpendicular1
        length = math.sqrt(sum(x*x for x in perpendicular1))
        if length > 0:
            perpendicular1 = [x/length * head_width for x in perpendicular1]
        
        # Berechne zweiten senkrechten Vektor
        perpendicular2 = [
            direction[1] * perpendicular1[2] - direction[2] * perpendicular1[1],
            direction[2] * perpendicular1[0] - direction[0] * perpendicular1[2],
            direction[0] * perpendicular1[1] - direction[1] * perpendicular1[0]
        ]
        
        # Basis der Pfeilspitze
        base_point = [
            end_point[0] - direction[0] * head_length,
            end_point[1] - direction[1] * head_length,
            end_point[2] - direction[2] * head_length
        ]
        
        # 4 Punkte für die Pfeilspitze (Pyramide)
        tip_points = [
            [base_point[0] + perpendicular1[0], base_point[1] + perpendicular1[1], base_point[2] + perpendicular1[2]],
            [base_point[0] - perpendicular1[0], base_point[1] - perpendicular1[1], base_point[2] - perpendicular1[2]], 
            [base_point[0] + perpendicular2[0], base_point[1] + perpendicular2[1], base_point[2] + perpendicular2[2]],
            [base_point[0] - perpendicular2[0], base_point[1] - perpendicular2[1], base_point[2] - perpendicular2[2]]
        ]
        
        # Zeichne Pfeilspitze als Linien von der Spitze zu den Basispunkten
        glColor3f(1.0, 0.0, 0.0)  # Rot
        glBegin(GL_LINES)
        for tip_point in tip_points:
            glVertex3f(end_point[0], end_point[1], end_point[2])  # Pfeilspitze
            glVertex3f(tip_point[0], tip_point[1], tip_point[2])   # Basispunkt
        glEnd()
        
    def mousePressEvent(self, event):
        """Behandelt Mausklicks"""
        if event.button() == Qt.MouseButton.LeftButton:
            print(f"Linksklick bei Position: {event.position().x()}, {event.position().y()}")
            
            # Verwende Ray-Casting für präzise Punktauswahl
            point = self.mouse_to_3d(event.position().x(), event.position().y())
            print(f"Berechneter 3D-Punkt: {point}")
            
            if point is not None:
                # Einfache Lösung: Alle Anschlussvektoren zeigen in Z-Richtung des Weltkoordinatensystems
                world_z_direction = [0.0, 0.0, 1.0]
                print(f"Welt-Z-Richtung: {world_z_direction}")
                
                # Erstelle Vektor-Objekt
                vector_data = {
                    'start_point': point,
                    'direction': world_z_direction
                }
                
                self.selected_vectors.append(vector_data)
                print(f"Vektor hinzugefügt. Anzahl Vektoren: {len(self.selected_vectors)}")
                
                self.pointSelected.emit(point[0], point[1], point[2])
                self.update()
            else:
                print("Kein 3D-Punkt gefunden")
        
        self.last_mouse_pos = event.position()
        
    def mouseMoveEvent(self, event):
        """Behandelt Mausbewegung für Kamera-Rotation"""
        if event.buttons() == Qt.MouseButton.RightButton and self.last_mouse_pos:
            dx = event.position().x() - self.last_mouse_pos.x()
            dy = event.position().y() - self.last_mouse_pos.y()
            
            self.camera_rotation_y += dx * 0.5
            self.camera_rotation_x += dy * 0.5
            
            self.update()
            
        self.last_mouse_pos = event.position()
        
    def wheelEvent(self, event):
        """Behandelt Mausrad für Zoom"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 1/1.1
        
        self.camera_distance *= zoom_factor
        self.camera_distance = max(0.05, min(5.0, self.camera_distance))
        
        self.update()
        
    def mouse_to_3d(self, x, y):
        """OpenGL-basiertes Ray-Casting mit glReadPixels für Tiefenwert"""
        if self.mesh_vertices is None or self.mesh_faces is None:
            print("Fehler: Keine Mesh-Daten vorhanden")
            return None
            
        print(f"OpenGL Ray-Casting für Mausposition ({x}, {y})")
        
        # Stelle sicher dass das Widget den aktuellen OpenGL-Kontext hat
        self.makeCurrent()
        
        # Lese Tiefenwert an der Mausposition
        # Y-Koordinate für OpenGL umkehren (OpenGL hat Y=0 unten)
        gl_y = self.height() - y
        
        try:
            # Lese Tiefenwert aus dem Depth Buffer
            depth = glReadPixels(int(x), int(gl_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]
            print(f"Depth-Wert an Position ({x}, {gl_y}): {depth}")
            
            # Wenn Tiefenwert 1.0 ist, wurde kein Objekt getroffen
            if depth >= 1.0:
                print("Kein Objekt an dieser Position (depth = 1.0)")
                return None
                
            # Hole OpenGL Matrizen
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX) 
            viewport = glGetIntegerv(GL_VIEWPORT)
            
            # Verwende gluUnProject mit dem gelesenen Tiefenwert
            from OpenGL.GLU import gluUnProject
            
            world_x, world_y, world_z = gluUnProject(
                x, gl_y, depth,
                modelview, projection, viewport
            )
            
            world_point = [world_x, world_y, world_z]
            print(f"3D Weltkoordinaten: {world_point}")
            
            # Transformiere zurück zu ursprünglichen Koordinaten
            if hasattr(self, 'mesh_center') and hasattr(self, 'mesh_scale'):
                original_point = [
                    world_point[0] * self.mesh_scale + self.mesh_center[0],
                    world_point[1] * self.mesh_scale + self.mesh_center[1], 
                    world_point[2] * self.mesh_scale + self.mesh_center[2]
                ]
                print(f"Transformiert zu ursprünglichen Koordinaten: {original_point}")
                return original_point
            else:
                return world_point
                
        except Exception as e:
            print(f"Fehler beim OpenGL Ray-Casting: {e}")
            return None
        
    def ray_triangle_intersect(self, ray_origin, ray_direction, v0, v1, v2):
        """Möller-Trumbore Ray-Triangle Intersection Algorithm - Robuste Version"""
        EPSILON = 1e-6
        
        try:
            # Kanten des Dreiecks
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            # Cross product von Ray-Richtung und edge2
            h = np.cross(ray_direction, edge2)
            
            # Determinante
            det = np.dot(edge1, h)
            
            # Ray ist parallel zum Dreieck
            if abs(det) < EPSILON:
                return None
                
            inv_det = 1.0 / det
            
            # Vektor vom Eckpunkt zum Ray-Ursprung
            s = ray_origin - v0
            
            # Baryzentrischer Koordinate u
            u = inv_det * np.dot(s, h)
            if u < 0.0 or u > 1.0:
                return None
                
            # Cross product von s und edge1
            q = np.cross(s, edge1)
            
            # Baryzentrischer Koordinate v
            v = inv_det * np.dot(ray_direction, q)
            if v < 0.0 or u + v > 1.0:
                return None
                
            # Berechne t (Distanz entlang Ray)
            t = inv_det * np.dot(edge2, q)
            
            # Ray intersection - erweitere den zulässigen Bereich
            if t > EPSILON and t < 1000.0:  # Erlaube größere Distanzen
                intersection_point = ray_origin + t * ray_direction
                return intersection_point
                
            return None
            
        except Exception as e:
            print(f"Fehler im Ray-Triangle-Test: {e}")
            return None
        
    def get_camera_direction(self):
        """Berechnet die normalisierte Richtung vom Objekt zur Kamera (für Normalvektoren)"""
        import math
        
        # Konvertiere Rotation zu Radianten
        rad_x = math.radians(self.camera_rotation_x)
        rad_y = math.radians(self.camera_rotation_y)
        
        # Berechne Kamera-Position relativ zum Objekt
        cam_x = self.camera_distance * math.sin(rad_y) * math.cos(rad_x)
        cam_y = -self.camera_distance * math.sin(rad_x)
        cam_z = self.camera_distance * math.cos(rad_y) * math.cos(rad_x)
        
        # Richtung vom Objekt (0,0,0) zur Kamera
        direction_x = cam_x
        direction_y = cam_y  
        direction_z = cam_z
        
        # Normalisiere den Vektor
        length = math.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
        if length > 0:
            direction_x /= length
            direction_y /= length
            direction_z /= length
        
        return [direction_x, direction_y, direction_z]

    def get_camera_position(self):
        """Berechnet die absolute Position der Kamera im Weltkoordinatensystem"""
        import math
        
        rad_x = math.radians(self.camera_rotation_x)
        rad_y = math.radians(self.camera_rotation_y)
        
        # Kamera-Position in ursprünglichen Koordinaten (vor Normalisierung)
        if hasattr(self, 'mesh_center') and hasattr(self, 'mesh_scale'):
            # Berechne Position in normalisierten Koordinaten
            cam_x_norm = self.camera_distance * math.sin(rad_y) * math.cos(rad_x)
            cam_y_norm = -self.camera_distance * math.sin(rad_x)
            cam_z_norm = self.camera_distance * math.cos(rad_y) * math.cos(rad_x)
            
            # Transformiere zurück zu ursprünglichen Koordinaten
            cam_x = cam_x_norm * self.mesh_scale + self.mesh_center[0]
            cam_y = cam_y_norm * self.mesh_scale + self.mesh_center[1]
            cam_z = cam_z_norm * self.mesh_scale + self.mesh_center[2]
            
            return [cam_x, cam_y, cam_z]
        else:
            # Fallback ohne Transformation
            cam_x = self.camera_distance * math.sin(rad_y) * math.cos(rad_x)
            cam_y = -self.camera_distance * math.sin(rad_x)
            cam_z = self.camera_distance * math.cos(rad_y) * math.cos(rad_x)
            return [cam_x, cam_y, cam_z]

    def get_view_direction(self):
        """Berechnet die Richtung orthogonal aus dem Bildschirm zum Benutzer"""
        # Einfache Lösung: Immer in positive Z-Richtung des View-Space
        # Das entspricht "aus dem Bildschirm heraus" zum Benutzer
        # Unabhängig von der Kamera-Rotation
        
        # In OpenGL View-Space zeigt die positive Z-Achse zum Betrachter
        # (entgegen der Standard-Konvention wo -Z nach vorne zeigt)
        return [0.0, 0.0, 1.0]

    def get_screen_normal_direction(self):
        """Berechnet die Richtung orthogonal zur Bildschirmebene (aus dem Bildschirm heraus)"""
        # Die Screen-Normal entspricht der inversen Kamera-Blickrichtung
        # In OpenGL zeigt die Kamera standardmäßig in -Z Richtung
        # Wir wollen aber +Z (aus dem Bildschirm heraus)
        
        import math
        rad_x = math.radians(self.camera_rotation_x)
        rad_y = math.radians(self.camera_rotation_y)
        
        # Berechne die Kamera-Forward-Richtung (wohin die Kamera blickt)
        # Und invertiere sie für "aus dem Bildschirm heraus"
        forward_x = math.sin(rad_y) * math.cos(rad_x)
        forward_y = -math.sin(rad_x)  # Negativ wegen der Rotation
        forward_z = math.cos(rad_y) * math.cos(rad_x)
        
        # Normalisiere
        length = math.sqrt(forward_x**2 + forward_y**2 + forward_z**2)
        if length > 0:
            forward_x /= length
            forward_y /= length
            forward_z /= length
            
        return [forward_x, forward_y, forward_z]

    def get_current_view_normal(self):
        """Transformiert Camera-Space Z-Vektor [0,0,1] zu World-Space mit 3x3 Matrix"""
        import math
        import numpy as np
        
        # Camera-Space Z-Vektor (zeigt aus dem Bildschirm heraus)
        camera_z = np.array([0.0, 0.0, 1.0])
        
        # Rotationswinkel 
        rad_x = math.radians(self.camera_rotation_x)
        rad_y = math.radians(self.camera_rotation_y)
        
        # 3x3 Rotationsmatrix um X-Achse
        rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rad_x), -math.sin(rad_x)],
            [0.0, math.sin(rad_x), math.cos(rad_x)]
        ])
        
        # 3x3 Rotationsmatrix um Y-Achse  
        rot_y = np.array([
            [math.cos(rad_y), 0.0, math.sin(rad_y)],
            [0.0, 1.0, 0.0],
            [-math.sin(rad_y), 0.0, math.cos(rad_y)]
        ])
        
        # Kombinierte Rotationsmatrix (erst Y, dann X - wie in OpenGL)
        # Für die Inverse (Camera-zu-World) transponieren wir die Matrix
        camera_to_world = (rot_y @ rot_x).T
        
        # Transformiere Camera-Z-Vektor zu World-Space
        world_normal = camera_to_world @ camera_z
        
        # Normalisiere (sollte bereits normalisiert sein)
        length = np.linalg.norm(world_normal)
        if length > 0:
            world_normal = world_normal / length
        
        return world_normal.tolist()

    def get_surface_point_simplified(self, mouse_x, mouse_y):
        """Vereinfachte Methode um einen Oberflächenpunkt zu bestimmen"""
        if self.mesh_vertices is None or len(self.mesh_vertices) == 0:
            return None
        
        # Berechne Offset basierend auf Mausposition (relativ zum Zentrum)
        widget_width = self.width()
        widget_height = self.height()
        
        # Normalisierte Mausposition (-1 bis 1)
        normalized_x = (mouse_x / widget_width) * 2.0 - 1.0
        normalized_y = (mouse_y / widget_height) * 2.0 - 1.0
        
        # Invertiere Y (OpenGL Koordinatensystem)
        normalized_y = -normalized_y
        
        # Erstelle Punkt leicht versetzt vom Objektzentrum
        # Das normalisierte Objekt ist um [0,0,0] zentriert
        base_point = [0.0, 0.0, 0.0]  # Objektzentrum
        
        # Füge kleine Verschiebung basierend auf Mausposition hinzu
        offset_scale = 0.3  # Maximaler Offset
        point = [
            base_point[0] + normalized_x * offset_scale,
            base_point[1] + normalized_y * offset_scale,  
            base_point[2]  # Z bleibt am Zentrum
        ]
        
        print(f"Vereinfachter Punkt erstellt: {point} (Maus: {normalized_x:.3f}, {normalized_y:.3f})")
        return point
        
    def clear_points(self):
        """Löscht alle ausgewählten Vektoren"""
        self.selected_vectors.clear()
        self.update()
        
    def get_points(self):
        """Gibt alle ausgewählten Vektoren zurück"""
        return self.selected_vectors.copy()

    def set_mesh_visibility(self, visible):
        """Setzt die Sichtbarkeit des Mesh"""
        self.show_mesh = visible
        self.update()
        
    def keyPressEvent(self, event):
        """Behandelt Tastatureingaben für Zoom"""
        if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            # Zoom in
            self.camera_distance *= 0.9
            self.camera_distance = max(0.05, self.camera_distance)
            self.update()
        elif event.key() == Qt.Key.Key_Minus:
            # Zoom out
            self.camera_distance *= 1.1
            self.camera_distance = min(5.0, self.camera_distance)
            self.update()
        else:
            super().keyPressEvent(event)
            
    def zoom_in(self):
        """Zoom in per Funktionsaufruf"""
        self.camera_distance *= 0.9
        self.camera_distance = max(0.05, self.camera_distance)
        self.update()
        
    def zoom_out(self):
        """Zoom out per Funktionsaufruf"""
        self.camera_distance *= 1.1
        self.camera_distance = min(5.0, self.camera_distance)
        self.update()
        
    def reset_view(self):
        """Setzt die Kamera auf Standardansicht zurück"""
        self.camera_distance = 2.5
        self.camera_rotation_x = -20
        self.camera_rotation_y = 45
        self.update()

class PhoenixMainWindow(QMainWindow):
    """Hauptfenster der Phoenix Contact Anwendung"""
    
    def __init__(self):
        super().__init__()
        self.selected_points = []
        self.connection_points_file = "phoenix_connection_points.json"
        self.current_step_file = None  # Pfad zur aktuell geladenen STEP-Datei
        
        self.initUI()
        # Lade keine Datei automatisch - User wählt sie aus
        
    def initUI(self):
        """Initialisiert die Benutzeroberfläche"""
        self.setWindowTitle("Phoenix Contact PTI Terminal - Anschlusspunkt-Auswahl")
        self.setGeometry(100, 100, 1200, 800)
        
        # Zentrales Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        main_layout = QHBoxLayout(central_widget)
        
        # 3D-Viewer
        self.viewer = Phoenix3DViewer()
        self.viewer.pointSelected.connect(self.on_point_selected)
        main_layout.addWidget(self.viewer, stretch=3)
        
        # Rechte Seitenleiste
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        
        # Datei-Auswahl
        file_btn = QPushButton("📁 STEP-Datei öffnen")
        file_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        file_btn.clicked.connect(self.open_step_file)
        sidebar_layout.addWidget(file_btn)
        
        # Titel
        title_label = QLabel("Anschlussvektoren")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        sidebar_layout.addWidget(title_label)
        
        # Checkbox für Objektsichtbarkeit
        self.mesh_visibility_checkbox = QCheckBox("Objekt anzeigen")
        self.mesh_visibility_checkbox.setChecked(True)
        self.mesh_visibility_checkbox.toggled.connect(self.toggle_mesh_visibility)
        sidebar_layout.addWidget(self.mesh_visibility_checkbox)
        
        # Checkbox für Koordinatensystem
        self.coordinate_system_checkbox = QCheckBox("Koordinatensystem anzeigen")
        self.coordinate_system_checkbox.setChecked(True)
        self.coordinate_system_checkbox.toggled.connect(self.toggle_coordinate_system)
        sidebar_layout.addWidget(self.coordinate_system_checkbox)
        
        # Punktliste
        self.points_list = QListWidget()
        sidebar_layout.addWidget(self.points_list)
        
        # Zoom-Buttons
        zoom_layout = QHBoxLayout()
        btn_zoom_in = QPushButton("🔍+")
        btn_zoom_in.setToolTip("Zoom hinein")
        btn_zoom_in.clicked.connect(self.viewer.zoom_in)
        zoom_layout.addWidget(btn_zoom_in)
        
        btn_zoom_out = QPushButton("🔍-")
        btn_zoom_out.setToolTip("Zoom heraus")
        btn_zoom_out.clicked.connect(self.viewer.zoom_out)
        zoom_layout.addWidget(btn_zoom_out)
        
        btn_reset = QPushButton("🎯")
        btn_reset.setToolTip("Ansicht zurücksetzen")
        btn_reset.clicked.connect(self.viewer.reset_view)
        zoom_layout.addWidget(btn_reset)
        
        btn_fit = QPushButton("📐")
        btn_fit.setToolTip("Objekt anpassen")
        btn_fit.clicked.connect(self.fit_object_to_view)
        zoom_layout.addWidget(btn_fit)
        
        zoom_widget = QWidget()
        zoom_widget.setLayout(zoom_layout)
        sidebar_layout.addWidget(zoom_widget)
        
        # Buttons
        btn_clear = QPushButton("Alle löschen")
        btn_clear.clicked.connect(self.clear_points)
        sidebar_layout.addWidget(btn_clear)
        
        btn_save = QPushButton("Vektoren speichern")
        btn_save.clicked.connect(self.save_points)
        sidebar_layout.addWidget(btn_save)
        
        # Anleitung
        help_label = QLabel(
            "BEDIENUNG:\n\n"
            "• Linksklick: Vektor setzen\n"
            "• Rechte Maus + Ziehen: Drehen\n"
            "• 🔍+/🔍-: Zoomen\n"
            "• 🎯: Ansicht zurücksetzen\n"
            "• 📐: Objekt anpassen\n\n"
            "Vektoren zeigen zur Kamera\nund werden nummeriert (1, 2, 3, ...)"
        )
        help_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        sidebar_layout.addWidget(help_label)
        
        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar, stretch=1)
        
        # Statusbar
        self.statusBar().showMessage("Bereit - Öffnen Sie eine STEP-Datei um zu beginnen")
        
    def open_step_file(self):
        """Öffnet File-Dialog zum Auswählen einer STEP-Datei"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "STEP-Datei auswählen",
            "",
            "STEP-Dateien (*.stp *.step);;Alle Dateien (*.*)"
        )
        
        if file_path:
            self.current_step_file = file_path  # Speichere den Pfad der STEP-Datei
            self.load_step_file(file_path)
            # Nach dem Laden der STEP-Datei, prüfe auf entsprechende JSON-Datei
            self.auto_load_json_if_exists()
            
    def load_step_file(self, step_file_path):
        """Lädt die STEP-Datei und konvertiert sie"""
        if not os.path.exists(step_file_path):
            QMessageBox.warning(self, "Fehler", f"STEP-Datei nicht gefunden: {step_file_path}")
            return
            
        try:
            self.statusBar().showMessage("Lade STEP-Datei...")
            
            # Lade mit trimesh
            phoenix_mesh = trimesh.load(step_file_path)
            
            if hasattr(phoenix_mesh, 'dump'):
                meshes = list(phoenix_mesh.dump())
                if meshes:
                    first_mesh = meshes[0]
                    vertices = first_mesh.vertices
                    faces = first_mesh.faces
                    
                    # Lösche alte Punkte
                    self.clear_points()
                    
                    # Lade in 3D-Viewer
                    self.viewer.load_mesh_data(vertices, faces)
                    
                    # Update Window-Titel mit Dateinamen
                    filename = os.path.basename(step_file_path)
                    self.setWindowTitle(f"3D Viewer - {filename}")
                    
                    self.statusBar().showMessage(
                        f"{filename} geladen - "
                        f"{len(vertices)} Vertices, {len(faces)} Faces"
                    )
                    
            elif hasattr(phoenix_mesh, 'vertices'):
                # Direktes Mesh ohne Scene
                vertices = phoenix_mesh.vertices
                faces = phoenix_mesh.faces
                
                # Lösche alte Punkte
                self.clear_points()
                
                self.viewer.load_mesh_data(vertices, faces)
                
                filename = os.path.basename(step_file_path)
                self.setWindowTitle(f"3D Viewer - {filename}")
                
                self.statusBar().showMessage(
                    f"{filename} geladen - "
                    f"{len(vertices)} Vertices, {len(faces)} Faces"
                )
            else:
                QMessageBox.warning(self, "Fehler", "Keine gültige 3D-Geometrie in der Datei gefunden")
                    
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Laden der STEP-Datei:\n{str(e)}")
            self.statusBar().showMessage("Fehler beim Laden der Datei")
            
    def auto_load_json_if_exists(self):
        """Lädt automatisch JSON-Vektoren wenn entsprechende Datei existiert"""
        if not self.current_step_file:
            return
            
        # Bestimme JSON-Dateiname basierend auf STEP-Datei
        step_dir = os.path.dirname(self.current_step_file)
        step_filename = os.path.basename(self.current_step_file)
        step_name_without_ext = os.path.splitext(step_filename)[0]
        json_filename = f"{step_name_without_ext}.json"
        json_filepath = os.path.join(step_dir, json_filename)
        
        # Prüfe ob JSON-Datei existiert
        if os.path.exists(json_filepath):
            try:
                self.load_json_vectors(json_filepath)
                self.statusBar().showMessage(f"Vektoren aus {json_filename} automatisch geladen")
            except Exception as e:
                print(f"Fehler beim automatischen Laden der JSON-Datei: {e}")
                self.statusBar().showMessage(f"Warnung: Fehler beim Laden von {json_filename}")
        else:
            print(f"Keine entsprechende JSON-Datei gefunden: {json_filepath}")
            
    def load_json_vectors(self, json_filepath):
        """Lädt Vektoren aus JSON-Datei"""
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Lösche alte Vektoren
        self.clear_points()
        
        # Lade Vektoren
        if 'connection_vectors' in data:
            vectors = data['connection_vectors']
            
            for vector_data in vectors:
                # Erstelle Vektor-Objekt für den Viewer
                viewer_vector = {
                    'start_point': [
                        vector_data['position']['x'],
                        vector_data['position']['y'], 
                        vector_data['position']['z']
                    ],
                    'direction': [
                        vector_data['direction']['x'],
                        vector_data['direction']['y'],
                        vector_data['direction']['z']
                    ]
                }
                
                # Füge zum Viewer hinzu
                self.viewer.selected_vectors.append(viewer_vector)
                
                # Füge zu selected_points hinzu (für das Speichern)
                self.selected_points.append(vector_data)
                
                # Füge zur UI-Liste hinzu
                pos = vector_data['position']
                dir_vec = vector_data['direction']
                self.points_list.addItem(
                    f"Vektor {vector_data['id']}: "
                    f"[{pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f}] → "
                    f"[{dir_vec['x']:.3f}, {dir_vec['y']:.3f}, {dir_vec['z']:.3f}]"
                )
            
            # Aktualisiere die Anzeige
            self.viewer.update()
            
            print(f"Automatisch {len(vectors)} Vektoren aus JSON geladen")
            
    def fit_object_to_view(self):
        """Passt das aktuelle Objekt optimal ins Fenster"""
        if hasattr(self.viewer, 'mesh_vertices') and self.viewer.mesh_vertices is not None:
            self.viewer.fit_object_to_view()
            self.viewer.update()
            self.statusBar().showMessage("Objekt an Fenster angepasst")
            
    def on_point_selected(self, x, y, z):
        """Wird aufgerufen wenn ein Vektor ausgewählt wurde"""
        vector_id = len(self.selected_points) + 1
        
        # Hole den letzten ausgewählten Vektor vom Viewer
        if self.viewer.selected_vectors:
            last_vector = self.viewer.selected_vectors[-1]
            direction = last_vector['direction']
            
            vector_data = {
                "id": vector_id,
                "position": {"x": float(x), "y": float(y), "z": float(z)},
                "direction": {"x": float(direction[0]), "y": float(direction[1]), "z": float(direction[2])}
            }
            
            self.selected_points.append(vector_data)
            
            # Füge zur Liste hinzu
            self.points_list.addItem(f"Vektor {vector_id}: [{x:.4f}, {y:.4f}, {z:.4f}] → [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
            
            self.statusBar().showMessage(f"Vektor {vector_id} hinzugefügt - {len(self.selected_points)} Vektoren total")
        
    def clear_points(self):
        """Löscht alle Vektoren"""
        self.selected_points.clear()
        self.points_list.clear()
        self.viewer.clear_points()
        self.statusBar().showMessage("Alle Vektoren gelöscht")
        
    def save_points(self):
        """Speichert die Vektoren in JSON-Datei mit gleichem Namen wie STEP-Datei"""
        if not self.selected_points:
            QMessageBox.information(self, "Info", "Keine Vektoren zum Speichern vorhanden")
            return
            
        if not self.current_step_file:
            QMessageBox.warning(self, "Warnung", "Keine STEP-Datei geladen. Bitte zuerst eine STEP-Datei öffnen.")
            return
            
        try:
            # Bestimme JSON-Dateiname basierend auf STEP-Datei
            step_dir = os.path.dirname(self.current_step_file)
            step_filename = os.path.basename(self.current_step_file)
            step_name_without_ext = os.path.splitext(step_filename)[0]
            json_filename = f"{step_name_without_ext}.json"
            json_filepath = os.path.join(step_dir, json_filename)
            
            # Erstelle JSON-Daten
            data = {
                "source_step_file": step_filename,
                "connection_vectors": self.selected_points
            }
            
            # Speichere JSON-Datei
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            QMessageBox.information(
                self, "Gespeichert", 
                f"{len(self.selected_points)} Anschlussvektoren wurden in "
                f"'{json_filepath}' gespeichert"
            )
            
            self.statusBar().showMessage(f"{len(self.selected_points)} Vektoren in {json_filename} gespeichert")
            
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern:\n{str(e)}")

    def toggle_mesh_visibility(self, checked):
        """Schaltet die Sichtbarkeit des Mesh um"""
        self.viewer.set_mesh_visibility(checked)
        status = "sichtbar" if checked else "ausgeblendet"
        self.statusBar().showMessage(f"Objekt {status}")

    def toggle_coordinate_system(self, checked):
        """Schaltet die Sichtbarkeit des Koordinatensystems um"""
        self.viewer.show_coordinate_system = checked
        self.viewer.update()
        status = "sichtbar" if checked else "ausgeblendet"
        self.statusBar().showMessage(f"Koordinatensystem {status}")

def main():
    app = QApplication(sys.argv)
    
    # Anwendungsstil setzen
    app.setStyle('Fusion')  # Moderner Look
    
    window = PhoenixMainWindow()
    window.show()
    
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())