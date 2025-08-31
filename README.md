# STEP-Anschlussengineering: Automatische Kammer-Öffnungs-Erkennung

Dieses Repository enthält spezialisierte Python-Anwendungen für das **Anschlussengineering** - die präzise Erkennung und Analyse von Kammer-Öffnungen in 3D-CAD-Bauteilen. Das Hauptziel ist die **automatische Lokalisierung von Anschlussstellen** und deren **geometrischen Kammern** für Engineering-Anwendungen.

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

### 2. **connection_chamber_analyzer.py** - Intelligente Kammer-Öffnungs-Erkennung
Das Herzstück des Anschlussengineerings: Vollautomatische Erkennung von Kammer-Öffnungen und deren geometrischen Eigenschaften.

**🎯 Hauptziel: Kammer-Öffnungs-Erkennung**
- **Automatische Öffnungs-Detektion**: Erkennt Kammer-Eingangsöffnungen in STEP-Bauteilen
- **3D-Kammer-Mittelpunkte**: Berechnet präzise Mittelpunkte und Tiefen der erkannten Kammern
- **Anschluss-Zuordnung**: Ordnet jeden Anschlusspunkt seiner entsprechenden Kammer zu
- **Geometrische Analyse**: Liefert quantitative Daten für Engineering-Berechnungen

**🔧 Technische Funktionen:**
- **Voxel-basierte Tiefenbild-Analyse**: Hochauflösende 3D-zu-2D-Projektion (800x Resolution)
- **Kontur-Erkennung mit OpenCV**: Robuste Identifikation von Kammer-Umrissen mittels Gradientenanalyse  
- **Intelligente Kantenvervollständigung**: Schließt abgeschnittene Kammer-Konturen automatisch
- **Hierarchische Filterung**: Eliminiert verschachtelte Bereiche zur Fokussierung auf Hauptkammern

**📊 Engineering-Ausgaben:**
- **4-Panel Visualisierung**: Vollständige Analyse-Pipeline von Tiefenbild bis Kammer-Zuordnung
- **3D-Koordinaten**: Präzise X/Y/Z-Koordinaten der Kammer-Mittelpunkte und -Tiefen
- **JSON-Integration**: Erweitert Anschlussdaten um `chamber_center` für nahtlose Weiterverwendung
- **Batch-Verarbeitung**: Automatische Analyse ganzer STEP-Datei-Bibliotheken

## 💾 Anschlussengineering-Datenstrukturen
Das System generiert erweiterte JSON-Daten mit automatisch erkannten Kammer-Mittelpunkten:

```json
{
  "connection_vectors": [
    {
      "id": 1,
      "position": {"x": 0.002147, "y": 0.043655, "z": 0.017129},
      "direction": {"x": 0.0, "y": 0.0, "z": 1.0},
      "chamber_center": {"x": 0.002822, "y": 0.042836, "z": 0.035235}
    },
    {
      "id": 2, 
      "position": {"x": 0.002242, "y": 0.004457, "z": 0.018983},
      "direction": {"x": 0.0, "y": 0.0, "z": 1.0},
      "chamber_center": {"x": 0.002822, "y": 0.005582, "z": 0.035235}
    }
  ]
}
```

**Datenerklärung:**
- **`position`**: Anschlusspunkt-Koordinaten (manuell definiert)
- **`direction`**: Richtungsvektor des Anschlusses
- **`chamber_center`**: **Automatisch erkannter Kammer-Mittelpunkt** mit maximaler Tiefe
- **Engineering-Nutzen**: Vollständige geometrische Definition für Anschluss-Berechnungen

## 🚀 Anschlussengineering-Workflow

### 1. Anschlusspunkt-Definition mit step_3d_viewer.py
```bash
python step_3d_viewer.py
```
**Interaktive Anschlusspunkt-Erstellung:**
- Lädt STEP-Bauteile und ermöglicht das Klicken auf Anschlussstellen
- Generiert automatisch Richtungsvektoren basierend auf der Kameraposition  
- Exportiert Anschlussdaten als JSON zur Weiterverarbeitung

### 2. Automatische Kammer-Öffnungs-Erkennung mit connection_chamber_analyzer.py
```bash
# Batch-Analyse aller STEP-Dateien im Data-Ordner (empfohlen)
python connection_chamber_analyzer.py

# Einzelne STEP-Datei analysieren
python connection_chamber_analyzer.py model.stp
```
**Das Herzstück des Anschlussengineerings:**
- **Erkennt automatisch Kammer-Öffnungen** in den STEP-Bauteilen
- **Berechnet 3D-Kammer-Mittelpunkte** mit maximaler Tiefe (Z-Koordinate)
- **Erweitert JSON-Daten** um `chamber_center` für jeden Anschlusspunkt
- **Visualisiert Erkennungs-Pipeline** in detaillierten Analyse-Grafiken

**Generierte Ausgabedateien:**
- `{filename}_contours_analysis.png` - **Hauptergebnis**: 4-Panel Kammer-Erkennungs-Pipeline
- `{filename}_contours_filtered.png` - Finale erkannte Kammern ohne Verschachtelungen
- `{filename}_depth_image.png` - Hochauflösendes Tiefenbild (800x Resolution)
- `{filename}_gradient_magnitude.png` - Gradientenanalyse zur Kammer-Detektion
- **Erweiterte `{filename}.json`** - Anschlussdaten + automatisch erkannte Kammer-Mittelpunkte

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