# STEP 3D Viewer - WPF Application

Eine moderne WPF-Anwendung zur Visualisierung von 3D-Bauteilen und deren Anschlusspunkten aus JSON-Dateien.

## Features

- **3D-Visualisierung**: Darstellung von 3D-Meshes aus JSON-Dateien mit Points und Indices
- **Connection Points**: Visualisierung von Anschlusspunkten mit Kugeln und Richtungspfeilen
- **Interaktive Steuerung**:
  - Linke Maus + Ziehen: 3D-Modell drehen
  - Mausrad: Zoomen
  - Mittlere Maus + Ziehen: Verschieben
- **Bauteile-Liste**: Schnelle Auswahl aus allen verfügbaren JSON-Dateien im DataSet-Ordner
- **Sichtbarkeits-Optionen**:
  - 3D-Modell ein/ausblenden
  - Anschlusspunkte ein/ausblenden
  - Koordinatensystem ein/ausblenden

## Technologie-Stack

- **.NET 8.0** - Modernes .NET Framework
- **WPF** - Windows Presentation Foundation für die UI
- **HelixToolkit.Wpf 3.1.1** - 3D-Rendering und Kamera-Steuerung
- **Newtonsoft.Json** - JSON-Deserialisierung
- **Serilog 4.3.0** - Strukturiertes Logging mit File Sink und Debug Sink

## Projekt-Struktur

```
StepViewer/
├── Models/
│   └── DataModels.cs          # Datenmodelle (PartData, Graphic3D, ConnectionPoint, etc.)
├── Services/
│   └── DataService.cs         # Service zum Laden und Speichern von JSON-Dateien
├── MainWindow.xaml            # Haupt-UI-Definition
├── MainWindow.xaml.cs         # Code-Behind mit 3D-Rendering-Logik
└── App.xaml                   # Anwendungskonfiguration
```

## JSON-Datenformat

Die Anwendung liest JSON-Dateien basierend auf der Spezifikation in `DataSet.spec.json`:

```json
{
  "PartNr": "A-B.100-C09EJ01",
  "Graphic3d": {
    "Points": [
      { "X": -18.3, "Y": 2.5, "Z": 36.3 },
      ...
    ],
    "Indices": [0, 1, 2, ...]
  },
  "BoundingBox": {
    "Dimension": { "X": 100.0, "Y": 50.0, "Z": 30.0 },
    "Location": { "X": 0.0, "Y": 0.0, "Z": 0.0 }
  },
  "ConnectionPoints": [
    {
      "Index": 1,
      "Name": "Connection 1",
      "Point": { "X": 10.0, "Y": 5.0, "Z": 15.0 },
      "InsertDirection": { "X": 0.0, "Y": 0.0, "Z": 1.0 }
    }
  ],
  "ProductTopGroup": 1,
  "ProductGroup": 2,
  "ProductSubGroup": 3
}
```

## Installation & Build

### Voraussetzungen

- .NET 8 SDK oder höher
- Windows 10/11

### Build

```bash
cd StepViewer
dotnet restore
dotnet build
```

### Ausführen

```bash
dotnet run
```

Oder alternativ die kompilierte EXE ausführen:
```bash
.\bin\Debug\net8.0-windows\StepViewer.exe
```

## Verwendung

1. **Starten Sie die Anwendung**
2. **Wählen Sie ein Bauteil** aus der Liste auf der rechten Seite
3. **Navigieren Sie im 3D-Viewport** mit der Maus:
   - **Drehen**: Linke Maustaste gedrückt halten und ziehen
   - **Verschieben (Pan)**: Mittlere Maustaste gedrückt halten und ziehen
   - **Zoomen**:
     - Mausrad scrollen
     - Rechte Maustaste gedrückt halten und ziehen
   - **ViewCube**: Klicken Sie auf eine Seite für Standardansichten
4. **Verwenden Sie die Ansichtsoptionen**:
   - Schalten Sie verschiedene Elemente ein/aus (3D-Modell, Anschlüsse, Koordinatensystem)
   - Verwenden Sie die Zoom-Buttons für präzises Zoomen
   - Klicken Sie auf "🎯" um die Ansicht zurückzusetzen

## Logging

Die Anwendung verwendet **Serilog** für strukturiertes Logging:

- **Log-Dateien**: `<Anwendungsverzeichnis>/Logs/stepviewer-YYYYMMDD.log`
- **Rotation**: Täglich, 30 Tage Aufbewahrung, Max. 10 MB pro Datei
- **Debug Output**: Logs erscheinen auch im Visual Studio Output Fenster
- **Exception Handling**: Alle unbehandelten Exceptions werden geloggt

Detaillierte Informationen finden Sie in [LOGGING.md](LOGGING.md).

## Architektur-Details

### 3D-Rendering

- **Mesh-Generierung**: Die `Points` und `Indices` aus den JSON-Dateien werden in WPF `MeshGeometry3D`-Objekte konvertiert
- **Normale-Berechnung**: Automatische Berechnung von Vertex-Normalen für korrektes Lighting
- **Material**: Graues Diffuse-Material mit Specular-Highlights

### Connection Points

- **Kugeln**: Rote Kugeln markieren die exakte Position der Anschlusspunkte
- **Pfeile**: Blaue Pfeile zeigen die `InsertDirection`
- **Labels**: Text-Billboards zeigen Index und Name des Anschlusspunkts
- **Adaptive Größe**: Größe der Visualisierungselemente wird automatisch basierend auf der BoundingBox berechnet

### Daten-Service

Der `DataService` bietet:
- Automatisches Laden aller JSON-Dateien aus dem DataSet-Ordner
- Fehlerbehandlung und Validierung
- Extraktion von Metadaten für die Dateiliste

## Migration von Python zu WPF

Diese Anwendung ersetzt die ursprüngliche PyQt6-basierte `step_3d_viewer.py` mit folgenden Verbesserungen:

- **Bessere Performance**: Native WPF 3D-Rendering ohne OpenGL-Overhead
- **Moderne UI**: Windows-native Oberfläche mit WPF-Styling
- **Typsicherheit**: Stark typisierte C#-Modelle statt dynamischer Python-Dicts
- **Einfachere Wartung**: Klare Trennung von UI, Business-Logik und Daten

## Bekannte Einschränkungen

- Nur Windows-Unterstützung (WPF ist Windows-spezifisch)
- Sehr große Meshes (>100.000 Dreiecke) können Performance-Probleme verursachen
- Aktuell keine Bearbeitungs-Funktionen für Connection Points (nur Anzeige)

## Zukünftige Erweiterungen

- [ ] Click-to-Select für Connection Points
- [ ] Hinzufügen/Bearbeiten/Löschen von Connection Points
- [ ] Export-Funktionalität
- [ ] Undo/Redo-Funktionalität
- [ ] Mehrfach-Auswahl von Bauteilen
- [ ] Performance-Optimierungen für große Meshes (LOD, Instancing)

## Lizenz

Entsprechend dem übergeordneten Projekt.
