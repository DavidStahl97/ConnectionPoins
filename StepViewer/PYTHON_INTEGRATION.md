# Python.NET Integration - Chamber Analyzer

Die WPF-Anwendung integriert den Python-basierten **Chamber Analyzer** über **Python.NET 3.0.5** für automatische Kammer-Erkennung.

## Architektur

### Komponenten

1. **chamber_analyzer_json.py** (Python)
   - Liest JSON-Daten im neuen DataSet-Format
   - Konvertiert Mesh (Points + Indices) zu Voxeln mit Open3D
   - Erstellt Tiefenbild durch Z-Projektion
   - Erkennt Konturen mit OpenCV
   - Berechnet Kammer-Mittelpunkte

2. **PythonChamberAnalyzer.cs** (C# Service)
   - Initialisiert Python.NET Engine
   - Ruft Python-Funktionen auf
   - Konvertiert Ergebnisse zwischen Python und C#

3. **MainWindow (WPF UI)**
   - Button "🔬 Kammer analysieren"
   - Asynchrone Ausführung
   - Fortschrittsanzeige
   - Ergebnis-Visualisierung

## Datenfluss

```
JSON-Datei (DataSet Format)
    ↓
[WPF UI] Button-Click
    ↓
[C#] PythonChamberAnalyzer.AnalyzeChambers()
    ↓
[Python.NET] Py.Import("chamber_analyzer_json")
    ↓
[Python] analyze_chambers_from_json()
    ├── create_mesh_from_json()
    ├── mesh_to_voxels()
    ├── voxels_to_depth_image()
    ├── detect_contours_from_depth()
    └── calculate_chamber_centers()
    ↓
[Python] Return: {success, chamber_centers, contour_count}
    ↓
[C#] ChamberAnalysisResult
    ↓
[WPF UI] Anzeige & JSON-Update
```

## Python-Abhängigkeiten

Die Analyse benötigt folgende Python-Packages (aus `requirements.txt`):

```
numpy
trimesh
open3d
opencv-python
scipy
matplotlib
```

**Installation**:
```bash
# Aktiviere venv
venv\Scripts\activate

# Installiere Abhängigkeiten
pip install -r requirements.txt
```

## Python.NET Setup

### 1. NuGet Package

```xml
<PackageReference Include="pythonnet" Version="3.0.5" />
```

### 2. Python-Umgebung

**Bevorzugt: venv**
- Sucht nach `venv/` im Projektverzeichnis
- Setzt `PYTHONHOME` und `PYTHONPATH`

**Fallback: System-Python**
- Verwendet global installiertes Python
- Erfordert dass alle Packages global installiert sind

### 3. Python Engine Initialisierung

```csharp
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();
```

**Wichtig**:
- Initialisierung erfolgt beim ersten Aufruf (lazy)
- Thread-Safe durch Lock
- Shutdown beim Dispose

## Verwendung im Code

### C# Aufruf

```csharp
using var analyzer = new PythonChamberAnalyzer();

var result = analyzer.AnalyzeChambers(
    jsonFilePath: "DataSet/part.json",
    outputDir: "DataSet/part"  // Optional: für Visualisierungen
);

if (result.Success)
{
    foreach (var center in result.ChamberCenters)
    {
        Console.WriteLine($"{center.ConnectionPointName}: {center.Center}");
    }
}
```

### Python Standalone

Das Python-Skript kann auch standalone ausgeführt werden:

```bash
python chamber_analyzer_json.py DataSet/part.json DataSet/part
```

## Ergebnis-Format

### ChamberAnalysisResult (C#)

```csharp
public class ChamberAnalysisResult
{
    public bool Success { get; set; }
    public string ErrorMessage { get; set; }
    public List<ChamberCenter> ChamberCenters { get; set; }
    public int ContourCount { get; set; }
}

public class ChamberCenter
{
    public int ConnectionPointIndex { get; set; }
    public string ConnectionPointName { get; set; }
    public Point3DData? Center { get; set; }  // X, Y, Z oder null
}
```

### Python Dict

```python
{
    'success': True,
    'chamber_centers': [
        {
            'connection_point_index': 1,
            'connection_point_name': 'Connection 1',
            'chamber_center': {
                'X': 0.00282,
                'Y': 0.04283,
                'Z': 0.03523
            }
        }
    ],
    'contour_count': 4
}
```

## Algorithmus

### 1. Mesh zu Voxel (Resolution: 800)

```python
# Konvertiere Points/Indices zu trimesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Open3D Voxelisierung
voxel_size = max_dimension / 800
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
```

### 2. Tiefenbild-Projektion (Z-Richtung)

```python
# Projiziere Voxel auf XY-Ebene
depth_image[y_idx, x_idx] = max(z_coords)  # Maximale Tiefe
```

### 3. Kontur-Erkennung

```python
# Gradienten mit Sobel
grad_x = cv2.Sobel(depth, CV_64F, 1, 0)
grad_y = cv2.Sobel(depth, CV_64F, 0, 1)
gradient_magnitude = sqrt(grad_x² + grad_y²)

# Schwellwert & Konturen
binary = gradient_magnitude > threshold
contours = cv2.findContours(binary, RETR_TREE, CHAIN_APPROX_SIMPLE)
```

### 4. Kammer-Mittelpunkte

```python
# Finde enthaltende Kontur für jeden Connection Point
result = cv2.pointPolygonTest(contour, (cp_x, cp_y), False)

# Berechne Bounding Box Center
x, y, w, h = cv2.boundingRect(contour)
center_x = x + w / 2
center_y = y + h / 2

# Extrahiere Z aus Kontur-Rand-Pixeln
center_z = max(depth_image[contour_edge_pixels])
```

## Ausgabe-Dateien

Wenn `outputDir` angegeben wird, erstellt der Analyzer:

### 1. Tiefenbild
`{PartNr}_depth_image.png` - Farbkodierte Z-Werte

### 2. Kontur-Analyse
`{PartNr}_contours_analysis.png` - Zeigt:
- Tiefenbild (Hintergrund)
- Erkannte Konturen
- Connection Points (rot)
- Chamber Centers (grün)

## Fehlerbehandlung

### Python-Exceptions

```csharp
try
{
    var result = analyzer.AnalyzeChambers(jsonPath);
}
catch (PythonException pyEx)
{
    // Python-Fehler (ImportError, ValueError, etc.)
    Console.WriteLine($"Python error: {pyEx.Message}");
}
```

### Häufige Fehler

**1. Python nicht gefunden**
```
Error: Failed to initialize Python engine
```
**Lösung**: Installiere Python 3.x oder erstelle venv

**2. Packages fehlen**
```
Python error: ModuleNotFoundError: No module named 'open3d'
```
**Lösung**: `pip install -r requirements.txt`

**3. JSON-Format ungültig**
```
Python error: Missing required field in JSON: Graphic3d
```
**Lösung**: Prüfe JSON-Format gegen `DataSet.spec.json`

**4. Keine Konturen erkannt**
```
success: False
error: No contours detected
```
**Lösung**:
- Mesh zu einfach/klein
- Voxel-Resolution erhöhen
- Schwellwert anpassen

## Performance

### Typische Laufzeiten

| Mesh-Größe | Voxel-Resolution | Dauer |
|------------|------------------|-------|
| 1.000 Dreiecke | 800 | ~2 Sek. |
| 10.000 Dreiecke | 800 | ~5 Sek. |
| 100.000 Dreiecke | 800 | ~15 Sek. |

### Optimierung

**Async/Await in C#**:
```csharp
var result = await Task.Run(() =>
    analyzer.AnalyzeChambers(jsonPath)
);
```
- UI bleibt responsiv
- Keine Blockierung

**Voxel-Resolution reduzieren**:
```python
voxel_grid = mesh_to_voxels(mesh, voxel_resolution=400)  # Schneller, weniger präzise
```

## Logging

### C# (Serilog)

```
[INF] Analyzing chambers for file: DataSet/part.json
[DBG] Acquiring Python GIL
[DBG] Importing Python module
[DBG] Calling analyze_chambers_from_json
[INF] Chamber analysis successful: 4 centers found, 4 contours
```

### Python (stdout/stderr)

Python-Ausgaben werden zu C# weitergeleitet und geloggt:

```
Loaded: A-B.100-C09EJ01 - 1234 points, 4 connection points
Created mesh: 1234 vertices, 411 faces
Converting mesh to voxels (resolution: 800)
Voxel grid created: 15234 voxels
Depth image created: 823x654, 67.3% coverage
Detected 4 contours
```

## Thread-Safety

**Python GIL (Global Interpreter Lock)**:
```csharp
using (Py.GIL())
{
    // Nur ein Thread kann Python-Code ausführen
    var result = chamberAnalyzer.analyze_chambers_from_json(...);
}
```

**C# Lock**:
```csharp
lock (_lockObject)
{
    // Thread-Safe Initialisierung
    if (!_isInitialized)
    {
        PythonEngine.Initialize();
        _isInitialized = true;
    }
}
```

## Troubleshooting

### Debug-Schritte

1. **Python-Skript standalone testen**:
   ```bash
   python chamber_analyzer_json.py DataSet/test.json
   ```

2. **Python-Pfade prüfen**:
   ```csharp
   _logger.Information("Python script: {Path}", _pythonScriptPath);
   _logger.Information("Python home: {Home}", _pythonHome);
   ```

3. **GIL-Probleme**:
   - Nur ein `Py.GIL()` gleichzeitig
   - Immer `using` verwenden
   - Bei Deadlock: Python Engine neu starten

4. **Memory Leaks**:
   - Immer `Dispose()` aufrufen
   - Große numpy Arrays freigeben
   - Python Engine Shutdown bei Beenden

## Weiterführende Links

- **Python.NET Docs**: https://pythonnet.github.io/
- **Open3D Docs**: http://www.open3d.org/docs/
- **OpenCV Python**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **Trimesh Docs**: https://trimsh.org/

## Beispiel: Vollständiger Workflow

```csharp
// 1. Initialisiere Analyzer
using var analyzer = new PythonChamberAnalyzer();

// 2. Führe Analyse aus
var result = await Task.Run(() =>
{
    return analyzer.AnalyzeChambers(
        "C:\\DataSet\\part.json",
        "C:\\DataSet\\part_output"
    );
});

// 3. Verarbeite Ergebnis
if (result.Success)
{
    Console.WriteLine($"Found {result.ChamberCenters.Count} chambers");

    foreach (var cc in result.ChamberCenters)
    {
        if (cc.Center != null)
        {
            Console.WriteLine($"  {cc.ConnectionPointName}:");
            Console.WriteLine($"    Center: ({cc.Center.X:F4}, {cc.Center.Y:F4}, {cc.Center.Z:F4})");
        }
        else
        {
            Console.WriteLine($"  {cc.ConnectionPointName}: No chamber found");
        }
    }
}
else
{
    Console.WriteLine($"Analysis failed: {result.ErrorMessage}");
}

// 4. Analyzer wird automatisch disposed
```

## Lizenz-Hinweise

- **Python.NET**: MIT License
- **Open3D**: MIT License
- **OpenCV**: Apache 2.0 License
- **Trimesh**: MIT License

Alle verwendeten Libraries sind für kommerzielle Nutzung geeignet.
