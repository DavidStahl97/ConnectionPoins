import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
import os
import json

# %% [markdown]
# # Interaktive Python Entwicklung für 3D Datenverarbeitung
# 
# Diese Datei demonstriert interaktive Python-Entwicklung mit Open3D

# %%
print("Hallo aus dem interaktiven Modus!")
print("Bereit für 3D Datenverarbeitung mit Open3D")
print(f"Open3D Version: {o3d.__version__}")


# %%
# Phoenix Contact STEP-Datei laden mit Trimesh
step_file_path = "../pxc_3209510_24_04_PT-2-5_3D.stp"
print(f"Lade STEP-Datei: {step_file_path}")

if os.path.exists(step_file_path):
    try:
        # Lade STEP-Datei mit trimesh
        print("Lade STEP-Datei mit trimesh...")
        phoenix_mesh = trimesh.load(step_file_path)
        
        print(f"STEP-Datei erfolgreich geladen als: {type(phoenix_mesh)}")
        
        # STEP-Dateien werden oft als Scene geladen
        if hasattr(phoenix_mesh, 'dump'):
            # Extrahiere alle Meshes aus der Scene
            meshes = list(phoenix_mesh.dump())
            print(f"Scene enthaelt {len(meshes)} Meshes")
            
            if meshes:
                # Nimm das erste/groesste Mesh
                first_mesh = meshes[0]
                print(f"Verwende Mesh mit {len(first_mesh.vertices)} Vertices und {len(first_mesh.faces)} Faces")
                
                # Konvertiere trimesh zu Open3D
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(first_mesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(first_mesh.faces)
                
                # Berechne Normalen
                o3d_mesh.compute_vertex_normals()
                
                # Prüfe ob trimesh Farben hat
                if hasattr(first_mesh.visual, 'vertex_colors') and first_mesh.visual.vertex_colors is not None:
                    print("Vertex-Farben in STEP-Datei gefunden!")
                    colors = np.array(first_mesh.visual.vertex_colors)[:, :3] / 255.0  # RGB, normalisiert
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                elif hasattr(first_mesh.visual, 'face_colors') and first_mesh.visual.face_colors is not None:
                    print("Face-Farben in STEP-Datei gefunden!")
                    colors = np.array(first_mesh.visual.face_colors)[:, :3] / 255.0  # RGB, normalisiert
                    # Open3D braucht Vertex-Farben, also mitteln wir die Face-Farben
                    vertex_colors = np.zeros((len(first_mesh.vertices), 3))
                    for i, face in enumerate(first_mesh.faces):
                        vertex_colors[face] += colors[i]
                    # Normalisiere
                    vertex_counts = np.zeros(len(first_mesh.vertices))
                    for face in first_mesh.faces:
                        vertex_counts[face] += 1
                    vertex_colors = vertex_colors / vertex_counts[:, np.newaxis]
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                elif hasattr(first_mesh.visual, 'material') and first_mesh.visual.material is not None:
                    print("Material in STEP-Datei gefunden!")
                    material = first_mesh.visual.material
                    if hasattr(material, 'main_color') and material.main_color is not None:
                        color = material.main_color[:3] / 255.0
                        o3d_mesh.paint_uniform_color(color)
                        print(f"Material-Farbe angewendet: {color}")
                    else:
                        # Fallback: Phoenix Contact typisches Grau
                        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                        print("Keine Material-Farbe gefunden - verwende Grau")
                else:
                    print("Keine Farben in STEP-Datei gefunden - verwende Phoenix Contact Grau")
                    # Phoenix Contact typisches Grau
                    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                
                print("Mesh fuer Open3D konvertiert")
            else:
                print("Keine Meshes in der Scene gefunden")
                
        elif hasattr(phoenix_mesh, 'vertices'):
            # Direktes Mesh
            print(f"Direktes Mesh mit {len(phoenix_mesh.vertices)} Vertices und {len(phoenix_mesh.faces)} Faces")
            
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(phoenix_mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(phoenix_mesh.faces)
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
            
            print("Direktes Mesh fuer Open3D konvertiert")
            
        else:
            print("Geladenes Objekt ist kein gueltiges Mesh")
            
    except Exception as e:
        print(f"Fehler beim Laden der STEP-Datei: {e}")
        print("Moegliche Loesungen:")
        print("   - STEP-Datei ist beschaedigt")
        print("   - Zusaetzliche Dependencies erforderlich")
        phoenix_mesh = None
        
else:
    print(f"STEP-Datei nicht gefunden: {step_file_path}")
    phoenix_mesh = None

# %%
# Lade Anschlusspunkt-Daten aus JSON
connection_points_file = "phoenix_connection_points.json"
print(f"Lade Anschlusspunkt-Daten: {connection_points_file}")

connection_points = []
if os.path.exists(connection_points_file):
    with open(connection_points_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        connection_points = data.get('connection_points', [])
        
    print(f"Anschlusspunkte geladen: {len(connection_points)}")
    for point in connection_points:
        print(f"  - {point['id']} ({point['name']}): Position [{point['position']['x']}, {point['position']['y']}, {point['position']['z']}]")
else:
    print("Anschlusspunkt-Datei nicht gefunden")

# %%
# Erstelle 3D-Markierungen für Anschlusspunkte
geometries = []

# Füge die Phoenix-Klemme hinzu (falls vorhanden)
if 'o3d_mesh' in locals():
    geometries.append(o3d_mesh)
    print("Phoenix-Klemme zur Visualisierung hinzugefuegt")

# %%
# Einfache Visualisierung mit Punktauswahl
selected_points = []

# %%
# Plotly-basierte interaktive 3D-Visualisierung
import plotly.graph_objects as go
import plotly.offline as pyo

selected_points = []

def create_plotly_visualization(mesh):
    """Erstellt interaktive Plotly 3D-Visualisierung"""
    global selected_points
    selected_points = []
    
    # Konvertiere Open3D Mesh zu Plotly Format
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    print("Konvertiere Mesh zu Plotly Format...")
    print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")
    
    # Erstelle Plotly Mesh3d
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1], 
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightgray',
        opacity=0.8,
        name='Phoenix Contact Klemme'
    )
    
    # Erstelle Scatter Plot für ausgewählte Punkte
    points_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            symbol='circle'
        ),
        name='Anschlusspunkte',
        text=[],
        hovertemplate='Punkt %{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>'
    )
    
    # Layout konfigurieren
    layout = go.Layout(
        title='Phoenix Contact PTI Klemme - Klick für Anschlusspunkte',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text='<b>ANLEITUNG:</b><br>• Klicken Sie auf die graue Oberfläche<br>• Punkte werden automatisch nummeriert<br>• Schließen Sie das Fenster zum Speichern',
                showarrow=False,
                align='left',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        ]
    )
    
    # Erstelle Figure
    fig = go.Figure(data=[mesh_trace, points_trace], layout=layout)
    
    return fig

# JavaScript für Click-Events (wird in HTML eingebettet)
click_handler_js = """
<script>
var selectedPoints = [];
var pointCounter = 1;

document.addEventListener('DOMContentLoaded', function() {
    var gd = document.getElementById('plot');
    
    gd.on('plotly_click', function(data) {
        if (data.points.length > 0) {
            var point = data.points[0];
            
            // Nur auf Mesh klicken (nicht auf bereits gesetzte Punkte)
            if (point.curveNumber === 0) {
                var x = point.x;
                var y = point.y; 
                var z = point.z;
                
                console.log('Punkt ' + pointCounter + ' gesetzt: ', x, y, z);
                
                // Füge Punkt zur Liste hinzu
                selectedPoints.push({
                    id: pointCounter,
                    position: {x: x, y: y, z: z}
                });
                
                // Update Scatter-Plot
                var update = {
                    x: [selectedPoints.map(p => p.position.x)],
                    y: [selectedPoints.map(p => p.position.y)],
                    z: [selectedPoints.map(p => p.position.z)],
                    text: [selectedPoints.map(p => p.id)]
                };
                
                Plotly.restyle(gd, update, [1]);
                
                pointCounter++;
                
                // Speichere automatisch in lokalen Storage
                localStorage.setItem('selectedPoints', JSON.stringify(selectedPoints));
            }
        }
    });
    
    // Lade gespeicherte Punkte beim Start
    var saved = localStorage.getItem('selectedPoints');
    if (saved) {
        selectedPoints = JSON.parse(saved);
        pointCounter = selectedPoints.length + 1;
        
        if (selectedPoints.length > 0) {
            var update = {
                x: [selectedPoints.map(p => p.position.x)],
                y: [selectedPoints.map(p => p.position.y)],
                z: [selectedPoints.map(p => p.position.z)],
                text: [selectedPoints.map(p => p.id)]
            };
            Plotly.restyle(gd, update, [1]);
        }
    }
    
    // Speichern-Button hinzufügen
    var saveBtn = document.createElement('button');
    saveBtn.innerHTML = 'Punkte Speichern';
    saveBtn.style.position = 'fixed';
    saveBtn.style.top = '10px';
    saveBtn.style.right = '10px';
    saveBtn.style.padding = '10px';
    saveBtn.style.backgroundColor = '#007bff';
    saveBtn.style.color = 'white';
    saveBtn.style.border = 'none';
    saveBtn.style.borderRadius = '5px';
    saveBtn.style.cursor = 'pointer';
    saveBtn.onclick = function() {
        if (selectedPoints.length > 0) {
            // Download als JSON
            var dataStr = JSON.stringify(selectedPoints, null, 2);
            var dataBlob = new Blob([dataStr], {type: 'application/json'});
            var url = URL.createObjectURL(dataBlob);
            var link = document.createElement('a');
            link.href = url;
            link.download = 'selected_points.json';
            link.click();
            alert(selectedPoints.length + ' Punkte gespeichert!');
        } else {
            alert('Keine Punkte ausgewählt!');
        }
    };
    document.body.appendChild(saveBtn);
});
</script>
"""

if 'o3d_mesh' in locals():
    print("Starte Plotly-basierte 3D-Visualisierung...")
    
    # Erstelle Plotly-Visualisierung
    fig = create_plotly_visualization(o3d_mesh)
    
    # Speichere als HTML-Datei mit Click-Handler
    html_content = fig.to_html(include_plotlyjs=True)
    
    # Füge Click-Handler JavaScript hinzu
    html_content = html_content.replace('</head>', click_handler_js + '</head>')
    
    # Schreibe HTML-Datei
    html_file = 'phoenix_visualizer.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"3D-Visualisierung gespeichert als: {html_file}")
    print("\nANLEITUNG:")
    print("1. Öffnen Sie die Datei phoenix_visualizer.html in Ihrem Browser")
    print("2. Klicken Sie auf die graue Oberfläche der Klemme um Anschlusspunkte zu setzen")
    print("3. Rote Punkte zeigen Ihre Auswahl")
    print("4. Schließen Sie den Browser und drücken Sie hier Enter zum Speichern")
    
    # Öffne Browser automatisch
    import webbrowser
    import time
    
    print("Öffne Browser...")
    webbrowser.open(f'file://{os.path.abspath(html_file)}')
    
    input("\nDrücken Sie Enter wenn Sie mit der Punktauswahl fertig sind...")
    
    # Lese Punkte aus temporärer Datei
    temp_file = 'selected_points.json'
    selected_points = []
    
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                selected_points = json.load(f)
            os.remove(temp_file)  # Cleanup
            print(f"\n{len(selected_points)} Punkte aus Browser geladen")
        except Exception as e:
            print(f"Fehler beim Laden der Punkte: {e}")
    
    # Speichere Punkte in finale JSON-Datei
    if selected_points:
        print(f"Speichere {len(selected_points)} Anschlusspunkte...")
        
        data = {"connection_points": selected_points}
        
        with open(connection_points_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("Anschlusspunkte erfolgreich gespeichert!")
        
        for point in selected_points:
            print(f"  - Punkt {point['id']}: [{point['position']['x']:.4f}, {point['position']['y']:.4f}, {point['position']['z']:.4f}]")
    else:
        print("Keine Punkte ausgewählt")
    
else:
    print("Keine Phoenix-Klemme zum Anzeigen verfuegbar")