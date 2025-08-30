# Phoenix QT Viewer - ConnectionPoints

Der **phoenix_qt_viewer** ist eine PyQt6-basierte 3D-Viewer-Anwendung für Phoenix Contact Terminals mit folgenden Hauptfunktionen:

## 🔧 Kernfunktionalitäten
- **STEP-Datei Import**: Lädt und visualisiert 3D-CAD-Modelle (*.stp, *.step)
- **Interaktive 3D-Ansicht**: OpenGL-basierter Viewer mit Hardware-beschleunigtem Rendering
- **Anschlusspunkt-Auswahl**: Klickbare Oberflächenpunkte zur Definition von Anschlussvektoren

## 🎮 Benutzerinteraktion
- **Maussteuerung**: Linksklick für Punktauswahl, Rechtsklick+Ziehen für Kamera-Rotation
- **Zoom-Funktionen**: Mausrad, Tastatur (+/-) und GUI-Buttons
- **Kamera-Steuerung**: Automatische Objektanpassung und manuelle Ansichts-Zurücksetzung

## 📊 Datenverarbeitung
- **Vektor-Generierung**: Automatische Berechnung von Richtungsvektoren basierend auf Kameraposition
- **JSON-Export**: Speicherung der Anschlussvektoren mit Position und Richtung
- **Performance-Optimierung**: Display Lists und Vertex Arrays für flüssiges Rendering

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
Die Anwendung ermöglicht es Ingenieuren, 3D-Terminalmodelle zu laden und präzise Anschlusspunkte mit zugehörigen Richtungsvektoren interaktiv zu definieren.

## 💻 Ausführung
```bash
# Programm starten
cd ConnectionPoins
./venv/Scripts/python phoenix_qt_viewer.py
```

### Voraussetzungen
- Python 3.x mit aktiviertem virtuellen Environment
- Erforderliche Pakete: PyQt6, OpenGL, trimesh, open3d, numpy