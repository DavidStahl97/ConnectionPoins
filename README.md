# STEP 3D Viewer & Connection Chamber Analyzer - Connection Points

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

### 2. **connection_chamber_analyzer.py** - Anschlusspunkt-Kammer-Analyse
Kommandozeilen-basiertes Tool zur intelligenten Erkennung und Zuordnung von Anschlusspunkten zu geometrischen Kammern:

**🔧 Kernfunktionalitäten:**
- **Kammer-Erkennung**: Automatische Identifikation geschlossener Bereiche durch Kontur-Analyse
- **Anschlusspunkt-Zuordnung**: Präzise Bestimmung welcher Anschlusspunkt in welcher Kammer liegt
- **Intelligente Konturen-Vervollständigung**: Schließt abgeschnittene Kanten durch Rahmen-Erweiterung
- **Hierarchische Filterung**: Entfernt verschachtelte Konturen zur Fokussierung auf Hauptkammern

**📈 Analysefunktionen:**
- **4-Panel Visualisierung**: Vollständige Pipeline-Darstellung von Binärbild bis zur finalen Kammer-Zuordnung
- **OpenCV Kontur-Erkennung**: Robuste Identifikation zusammenhängender Bereiche mittels Gradientenanalyse
- **Batch-Verarbeitung**: Automatische Analyse aller STEP-Dateien im Data-Ordner
- **Detaillierte Statistiken**: Quantitative Auswertung der Kammer-Anschlusspunkt-Zuordnungen

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

### connection_chamber_analyzer.py
```bash
# Batch-Analyse aller STEP-Dateien im Data-Ordner
python connection_chamber_analyzer.py

# Mit einzelner STEP-Datei als Argument
python connection_chamber_analyzer.py model.stp
```
Analysiert die geometrischen Kammern in STEP-Dateien und ordnet Anschlusspunkte den entsprechenden Kammern zu. Erstellt dabei folgende Ausgabedateien:
- `{filename}_contours_analysis.png` - 4-Panel Visualisierung der Kammer-Analyse-Pipeline
- `{filename}_contours_filtered.png` - Finale Kammern ohne verschachtelte Bereiche
- `{filename}_depth_image.png` - Tiefenbild mit Anschlusspunkt-Markierungen
- `{filename}_gradient_magnitude.png` - Gradientenanalyse für Kammer-Erkennung

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

# Kammer-Analyzer für Batch-Verarbeitung ausführen
python connection_chamber_analyzer.py

# Oder einzelne STEP-Datei analysieren
python connection_chamber_analyzer.py [dateiname.stp]
```

### Voraussetzungen
- Python 3.8 oder höher
- Windows, Linux oder macOS
- OpenGL-fähige Grafikkarte