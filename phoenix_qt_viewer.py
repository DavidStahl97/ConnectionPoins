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
                             QListWidget, QMessageBox, QStatusBar, QFileDialog)
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
        
        # Zeichne Mesh mit Display List für bessere Performance
        self.draw_mesh_optimized()
        
        # Zeichne ausgewählte Vektoren als Pfeile
        self.draw_vectors()
        
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
        """Zeichnet die ausgewählten Vektoren als Würfel"""
        if not self.selected_vectors:
            return
            
        glDisable(GL_LIGHTING)
        
        for i, vector_data in enumerate(self.selected_vectors):
            start_point = vector_data['start_point']
            
            # Zeichne einen kleinen Würfel am Startpunkt
            self.draw_cube(start_point, 0.02)  # Würfel mit 2cm Kantenlänge
        
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
        
    def draw_arrow_head(self, start_point, end_point, direction):
        """Zeichnet eine Pfeilspitze"""
        import math
        
        # Pfeilspitze Parameter
        head_length = 0.02
        head_width = 0.01
        
        # Berechne senkrechte Vektoren für Pfeilspitze
        # Verwende Cross-Product mit einem beliebigen Vektor
        if abs(direction[1]) < 0.9:
            perpendicular1 = [
                direction[1] * 0 - direction[2] * 1,
                direction[2] * 1 - direction[0] * 0,
                direction[0] * 1 - direction[1] * 0
            ]
        else:
            perpendicular1 = [
                direction[1] * 1 - direction[2] * 0,
                direction[2] * 0 - direction[0] * 1,
                direction[0] * 0 - direction[1] * 1
            ]
        
        # Normalisiere
        length = math.sqrt(sum(x*x for x in perpendicular1))
        if length > 0:
            perpendicular1 = [x/length for x in perpendicular1]
        
        # Zweiter senkrechter Vektor
        perpendicular2 = [
            direction[1] * perpendicular1[2] - direction[2] * perpendicular1[1],
            direction[2] * perpendicular1[0] - direction[0] * perpendicular1[2],
            direction[0] * perpendicular1[1] - direction[1] * perpendicular1[0]
        ]
        
        # Rückwärts-Punkt für Pfeilspitze
        back_point = [
            end_point[0] - direction[0] * head_length,
            end_point[1] - direction[1] * head_length,
            end_point[2] - direction[2] * head_length
        ]
        
        # Pfeilspitze Eckpunkte
        tip_points = []
        for angle in [0, 120, 240]:  # 3 Ecken
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            
            tip_point = [
                back_point[0] + (perpendicular1[0] * cos_a + perpendicular2[0] * sin_a) * head_width,
                back_point[1] + (perpendicular1[1] * cos_a + perpendicular2[1] * sin_a) * head_width,
                back_point[2] + (perpendicular1[2] * cos_a + perpendicular2[2] * sin_a) * head_width
            ]
            tip_points.append(tip_point)
        
        # Zeichne Pfeilspitze als Linien
        glColor3f(1.0, 0.0, 0.0)  # Rot
        glBegin(GL_LINES)
        for tip_point in tip_points:
            glVertex3f(end_point[0], end_point[1], end_point[2])
            glVertex3f(tip_point[0], tip_point[1], tip_point[2])
        glEnd()
        
    def mousePressEvent(self, event):
        """Behandelt Mausklicks"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Konvertiere 2D-Mausposition zu 3D-Punkt
            point = self.mouse_to_3d(event.position().x(), event.position().y())
            if point is not None:
                # Berechne Kamera-Richtungsvektor
                camera_direction = self.get_camera_direction()
                
                # Erstelle Vektor-Objekt
                vector_data = {
                    'start_point': point,
                    'direction': camera_direction
                }
                
                self.selected_vectors.append(vector_data)
                self.pointSelected.emit(point[0], point[1], point[2])
                self.update()
        
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
        """Konvertiert 2D-Mausposition zu approximiertem 3D-Punkt auf Mesh"""
        # Vereinfachte Ray-Casting Approximation
        # In der Realität würde hier echtes Ray-Mesh-Intersection verwendet
        
        # Konvertiere zu normalisierte Koordinaten
        viewport = glGetIntegerv(GL_VIEWPORT)
        width, height = viewport[2], viewport[3]
        
        norm_x = (x - width/2) / (width/2)
        norm_y = (height/2 - y) / (height/2)  # Y ist invertiert
        
        # Projiziere auf eine Ebene vor der Kamera
        depth = 0.0  # Auf der Mesh-Oberfläche
        
        # Transformiere basierend auf aktueller Kamera
        scale = self.camera_distance * 0.1
        point_x = norm_x * scale
        point_y = norm_y * scale
        point_z = depth
        
        # Rotiere basierend auf Kamera-Rotation (vereinfacht)
        rad_y = math.radians(-self.camera_rotation_y)
        rad_x = math.radians(-self.camera_rotation_x)
        
        # Einfache Rotation um Y-Achse
        x_rot = point_x * math.cos(rad_y) - point_z * math.sin(rad_y)
        z_rot = point_x * math.sin(rad_y) + point_z * math.cos(rad_y)
        
        return [x_rot, point_y, z_rot]
        
    def get_camera_direction(self):
        """Berechnet die normalisierte Richtung von der Kamera zum Objekt"""
        import math
        
        # Konvertiere Rotation zu Radianten
        rad_x = math.radians(self.camera_rotation_x)
        rad_y = math.radians(self.camera_rotation_y)
        
        # Berechne Kamera-Richtungsvektor (invertiert da wir zur Kamera zeigen wollen)
        # Standard OpenGL Kamera blickt in negative Z-Richtung
        direction_x = math.sin(rad_y) * math.cos(rad_x)
        direction_y = math.sin(rad_x)
        direction_z = math.cos(rad_y) * math.cos(rad_x)
        
        # Normalisiere den Vektor
        length = math.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
        if length > 0:
            direction_x /= length
            direction_y /= length
            direction_z /= length
        
        return [direction_x, direction_y, direction_z]
        
    def clear_points(self):
        """Löscht alle ausgewählten Vektoren"""
        self.selected_vectors.clear()
        self.update()
        
    def get_points(self):
        """Gibt alle ausgewählten Vektoren zurück"""
        return self.selected_vectors.copy()
        
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
        self.camera_distance = 0.1
        self.camera_rotation_x = -20
        self.camera_rotation_y = 45
        self.update()

class PhoenixMainWindow(QMainWindow):
    """Hauptfenster der Phoenix Contact Anwendung"""
    
    def __init__(self):
        super().__init__()
        self.selected_points = []
        self.connection_points_file = "phoenix_connection_points.json"
        
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
            self.load_step_file(file_path)
            
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
        """Speichert die Vektoren in JSON-Datei"""
        if not self.selected_points:
            QMessageBox.information(self, "Info", "Keine Vektoren zum Speichern vorhanden")
            return
            
        try:
            data = {"connection_vectors": self.selected_points}
            
            with open(self.connection_points_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            QMessageBox.information(
                self, "Gespeichert", 
                f"{len(self.selected_points)} Anschlussvektoren wurden in "
                f"'{self.connection_points_file}' gespeichert"
            )
            
            self.statusBar().showMessage(f"{len(self.selected_points)} Vektoren gespeichert")
            
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern:\n{str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Anwendungsstil setzen
    app.setStyle('Fusion')  # Moderner Look
    
    window = PhoenixMainWindow()
    window.show()
    
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())