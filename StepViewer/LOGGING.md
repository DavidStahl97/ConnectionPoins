# Logging mit Serilog

Die Anwendung verwendet **Serilog** für strukturiertes Logging mit File Sink und Debug Sink.

## Konfiguration

### Log-Level

Die Logging-Konfiguration in `App.xaml.cs` verwendet folgende Log-Level:

- **Debug**: Detaillierte Informationen für Entwicklung und Debugging
- **Information**: Allgemeine Informationen über Anwendungsfluss
- **Warning**: Warnungen, die nicht zu Fehlern führen
- **Error**: Fehler, die zu Fehlfunktionen führen
- **Fatal**: Schwerwiegende Fehler, die zum Programmabsturz führen

### Log-Ausgaben

#### 1. File Sink (Datei-Logging)

Logs werden in rollende Dateien geschrieben:

**Speicherort**: `<Anwendungsverzeichnis>/Logs/`

**Dateiformat**: `stepviewer-YYYYMMDD.log`

**Eigenschaften**:
- Tägliches Rolling (neue Datei jeden Tag)
- Max. 30 Tage Aufbewahrung
- Max. Dateigröße: 10 MB
- Strukturiertes Format mit Timestamp, Level, Message und Exception

**Beispiel-Logzeile**:
```
2025-11-21 14:32:15.123 +01:00 [INF] Application starting up
2025-11-21 14:32:15.456 +01:00 [DBG] Loading file list from DataSet directory
2025-11-21 14:32:15.789 +01:00 [INF] Loaded 85 files from DataSet directory
```

#### 2. Debug Sink (Visual Studio Output)

Logs werden zusätzlich an das Debug-Fenster in Visual Studio gesendet:

**Format**: `HH:mm:ss [Level] Message`

**Beispiel**:
```
14:32:15 [INF] Application starting up
14:32:15 [DBG] Loading file list from DataSet directory
14:32:15 [INF] Loaded 85 files from DataSet directory
```

## Geloggte Events

### Application Lifecycle

```csharp
// App.xaml.cs
Log.Information("Application starting up");
Log.Information("Version: {Version}", version);
Log.Information("Log directory: {LogDirectory}", logDirectory);
Log.Information("Application shutting down with exit code: {ExitCode}", exitCode);
```

### Daten-Service (DataService.cs)

```csharp
// Initialisierung
Log.Information("DataService initialized with path: {DataSetPath}", path);

// Datei-Suche
Log.Debug("Searching for JSON files in: {DataSetPath}", path);
Log.Information("Found {FileCount} JSON files in DataSet directory", count);

// Datei-Laden
Log.Debug("Loading part data from: {FilePath}", filePath);
Log.Information("Successfully loaded part data: PartNr={PartNr}, Points={PointCount}, Connections={ConnectionCount}");

// Fehler
Log.Error(ex, "Invalid JSON format in file: {FilePath}", filePath);
Log.Warning(ex, "Failed to get file info for: {FilePath}", filePath);
```

### Main Window (MainWindow.xaml.cs)

```csharp
// Initialisierung
Log.Information("MainWindow initializing");
Log.Information("MainWindow initialized successfully");

// Datei-Liste laden
Log.Debug("Loading file list from DataSet directory");
Log.Information("Loaded {FileCount} files from DataSet directory", count);

// Part-Daten laden
Log.Information("Loading part data from file: {FilePath}", filePath);
Log.Debug("Part data loaded: PartNr={PartNr}, Points={PointCount}, Indices={IndexCount}, Connections={ConnectionCount}");

// 3D-Rendering
Log.Debug("Starting mesh rendering");
Log.Debug("Mesh rendered successfully with {TriangleCount} triangles", count);
Log.Debug("Rendering {ConnectionCount} connection points", count);

// UI-Interaktionen
Log.Debug("Zoom in requested");
Log.Debug("Reset view requested");

// Fehler
Log.Error(ex, "Failed to load part data from file: {FilePath}", filePath);
```

### Exception Handling

```csharp
// Unbehandelte Exceptions
Log.Fatal(exception, "Unhandled exception occurred. IsTerminating: {IsTerminating}", isTerminating);
Log.Error(exception, "Unhandled dispatcher exception occurred");
```

## Strukturiertes Logging

Serilog unterstützt strukturiertes Logging mit Named Properties:

```csharp
// ❌ String Interpolation (nicht strukturiert)
_logger.Information($"Loaded {count} files");

// ✅ Named Properties (strukturiert)
_logger.Information("Loaded {FileCount} files", count);
```

Strukturierte Logs können später einfach durchsucht und gefiltert werden.

## Log-Analyse

### Logs durchsuchen

**Windows PowerShell**:
```powershell
# Alle Fehler finden
Select-String -Path ".\Logs\*.log" -Pattern "\[ERR\]"

# Nach PartNr suchen
Select-String -Path ".\Logs\*.log" -Pattern "PartNr=A-B.100-C09EJ01"

# Logs von heute
Get-Content ".\Logs\stepviewer-$(Get-Date -Format 'yyyyMMdd').log"
```

**Visual Studio Output Fenster**:
- Debug > Windows > Output
- Filter auf "Debug" setzen
- Echtzeit-Logs während der Entwicklung

## Performance-Überlegungen

- **Asynchrones Logging**: Serilog verwendet asynchrones I/O für File Sink
- **Structured Data**: Named Properties sind effizienter als String Interpolation
- **Log-Level**: Production-Builds sollten `MinimumLevel.Information()` verwenden
- **File Rolling**: Alte Logs werden automatisch gelöscht (30 Tage Retention)
- **File Size Limit**: Einzelne Log-Dateien werden bei 10 MB rotiert

## Beispiel-Workflow

1. **Entwicklung**:
   - Logs erscheinen in Visual Studio Output Fenster
   - Debug-Level aktiviert für detaillierte Informationen

2. **Fehlersuche**:
   - Reproduziere das Problem
   - Öffne `Logs/stepviewer-YYYYMMDD.log`
   - Suche nach `[ERR]` oder `[FTL]` Einträgen
   - Prüfe Stack Traces und Context-Properties

3. **Production**:
   - Logs werden kontinuierlich in Dateien geschrieben
   - Alte Logs werden automatisch aufgeräumt
   - Bei Fehlern können Logs an Support-Team gesendet werden

## Integration in neue Komponenten

```csharp
using Serilog;

public class MyNewComponent
{
    private readonly ILogger _logger;

    public MyNewComponent()
    {
        // Logger mit Type Context erstellen
        _logger = Log.ForContext<MyNewComponent>();
    }

    public void DoSomething(string parameter)
    {
        _logger.Debug("DoSomething called with parameter: {Parameter}", parameter);

        try
        {
            // ... code ...
            _logger.Information("Operation completed successfully");
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Operation failed for parameter: {Parameter}", parameter);
            throw;
        }
    }
}
```

## Troubleshooting

### Logs werden nicht geschrieben

1. Prüfe ob `Logs/` Ordner existiert (wird automatisch erstellt)
2. Prüfe Schreibrechte im Anwendungsverzeichnis
3. Prüfe ob `Log.CloseAndFlush()` beim Beenden aufgerufen wird

### Zu viele Logs

- Erhöhe MinimumLevel von `Debug` auf `Information` in `App.xaml.cs`
- Reduziere Retention von 30 auf 7 Tage
- Reduziere File Size Limit

### Logs fehlen nach Crash

- Fatal Exceptions werden geloggt bevor die Anwendung beendet wird
- `Log.CloseAndFlush()` wird im `OnExit` Event aufgerufen
- Bei hard crashes (z.B. Kernel-Level) können Logs verloren gehen

## Weitere Informationen

- Serilog Dokumentation: https://serilog.net/
- Serilog.Sinks.File: https://github.com/serilog/serilog-sinks-file
- Structured Logging Best Practices: https://github.com/serilog/serilog/wiki/Structured-Data
