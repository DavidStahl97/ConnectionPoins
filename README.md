# STEP 3D Viewer & Voxel Converter - Connection Points

Dieses Repository enthält zwei spezialisierte Python-Anwendungen für die Arbeit mit STEP-Dateien und 3D-Geometrie-Analyse:

## 📋 Programme im Überblick

### 1. **step_3d_viewer.py** - Interaktive 3D-Visualisierung
PyQt6-basierte 3D-Viewer-Anwendung für STEP-Dateien mit folgenden Hauptfunktionen:

**🔧 Kernfunktionalitäten:**
- **STEP-Datei Import**: Lädt und visualisiert 3D-CAD-Modelle (*.stp, *.step)
- **Interaktive 3D-Ansicht**: OpenGL-basierter Viewer mit Hardware-beschleunigtem Rendering
- **Anschlusspunkt-Auswahl**: Klickbare Oberflächenpunkte zur Definition von Anschlussvektoren

**🎮 Benutzerinteraktion:**
- **Maussteuerung**: Linksklick für Punktauswahl, Rechtsklick+Ziehen für Kamera-Rotation
- **Zoom-Funktionen**: Mausrad, Tastatur (+/-) und GUI-Buttons
- **Kamera-Steuerung**: Automatische Objektanpassung und manuelle Ansichts-Zurücksetzung

**📊 Datenverarbeitung:**
- **Vektor-Generierung**: Automatische Berechnung von Richtungsvektoren basierend auf Kameraposition
- **JSON-Export**: Speicherung der Anschlussvektoren mit Position und Richtung
- **Performance-Optimierung**: Display Lists und Vertex Arrays für flüssiges Rendering

### 2. **step_to_voxel_converter.py** - Voxel-Konvertierung & Tiefenanalyse
Kommandozeilen-basiertes Tool für die Konvertierung von STEP-Dateien zu Voxel-Gittern und Tiefenbild-Analyse:

**🔧 Kernfunktionalitäten:**
- **STEP zu Voxel**: Konvertiert 3D-Meshes zu hochauflösenden Voxel-Gittern (400x400x400)
- **Tiefenbild-Generierung**: Erstellt 2D-Projektionen der Voxel-Daten
- **Gradientenanalyse**: Berechnet X/Y-Gradienten und Gradientenmagnitude mit OpenCV
- **Automatische Visualisierung**: matplotlib-basierte Darstellung aller Analyseergebnisse

**📈 Analysefunktionen:**
- **Sobel-Operatoren**: Präzise Gradientenberechnung für Oberflächenerkennung
- **Multi-Ansicht Visualisierung**: 2x2 Layout mit Original + 3 Gradientenbildern
- **Bildexport**: PNG-Speicherung aller Analyse-Ergebnisse
- **Anschlusspunkt-Integration**: Visualisierung der JSON-Vektordaten im Voxelraum

## 💾 Datenstrukturen
```json
{
  "connection_vectors": [
    {
      "id": 1,
      "position": {"x": 0.1234, "y": 0.5678, "z": 0.9012},
      "direction": {"x": 0.577, "y": 0.577, "z": 0.577}
    }
  ]
}
```

## 🚀 Verwendung

### step_3d_viewer.py
```bash
python step_3d_viewer.py
```
Ermöglicht es Ingenieuren, 3D-Modelle aus STEP-Dateien zu laden und präzise Anschlusspunkte mit zugehörigen Richtungsvektoren interaktiv zu definieren.

### step_to_voxel_converter.py
```bash
# Mit STEP-Datei als Argument
python step_to_voxel_converter.py model.stp

# Oder Standarddatei verwenden
python step_to_voxel_converter.py
```
Konvertiert STEP-Dateien zu Voxel-Gittern und führt automatisch Tiefenbild-Analyse mit Gradientenberechnung durch. Erstellt dabei folgende Ausgabedateien:
- `{filename}_depth_image.png` - Normalisiertes Tiefenbild
- `{filename}_gradient_magnitude.png` - Gradientenmagnitude für Oberflächenerkennung

## 🔧 Setup
```bash
# Repository klonen
git clone https://github.com/DavidStahl97/ConnectionPoins.git
cd ConnectionPoins

# Virtuelles Environment erstellen
python -m venv venv

# Environment aktivieren (Windows)
venv\Scripts\activate
# oder auf Linux/Mac: source venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt
```

## 💻 Ausführung
```bash
# Environment aktivieren (falls noch nicht aktiviert)
venv\Scripts\activate

# Interaktiver 3D-Viewer starten
python step_3d_viewer.py

# Oder Voxel-Konverter ausführen
python step_to_voxel_converter.py [dateiname.stp]
```

### Voraussetzungen
- Python 3.8 oder höher
- Windows, Linux oder macOS
- OpenGL-fähige Grafikkarte