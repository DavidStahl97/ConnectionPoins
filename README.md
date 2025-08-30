# STEP 3D Viewer - Connection Points

Der **step_3d_viewer** ist eine PyQt6-basierte 3D-Viewer-Anwendung für STEP-Dateien mit folgenden Hauptfunktionen:

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
Die Anwendung ermöglicht es Ingenieuren, 3D-Modelle aus STEP-Dateien zu laden und präzise Anschlusspunkte mit zugehörigen Richtungsvektoren interaktiv zu definieren.

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

# Programm starten
python step_3d_viewer.py
```

### Voraussetzungen
- Python 3.8 oder höher
- Windows, Linux oder macOS
- OpenGL-fähige Grafikkarte