using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Python.Runtime;
using Serilog;
using StepViewer.Models;

namespace StepViewer.Services
{
    /// <summary>
    /// Service for chamber analysis using Python.NET
    /// </summary>
    public class PythonChamberAnalyzer : IDisposable
    {
        private readonly ILogger _logger;
        private readonly string _pythonScriptPath;
        private readonly string? _pythonHome;
        private static bool _isInitialized = false;
        private static readonly object _lockObject = new object();

        public PythonChamberAnalyzer()
        {
            _logger = Log.ForContext<PythonChamberAnalyzer>();

            // Pfad zum Python-Skript (im Projektverzeichnis)
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var projectDir = Directory.GetParent(baseDir)?.Parent?.Parent?.Parent?.Parent?.FullName;
            _pythonScriptPath = Path.Combine(projectDir ?? "", "chamber_analyzer_json.py");

            // Python-Home (verwende venv wenn vorhanden)
            var venvPath = Path.Combine(projectDir ?? "", "venv");
            if (Directory.Exists(venvPath))
            {
                _pythonHome = venvPath;
                _logger.Information("Using Python venv: {VenvPath}", venvPath);
            }
            else
            {
                // Fallback: System-Python
                _pythonHome = null;
                _logger.Warning("No venv found, using system Python");
            }

            _logger.Information("Python script path: {ScriptPath}", _pythonScriptPath);

            // Setze Python DLL Pfad BEVOR die Engine initialisiert wird
            // Dies muss vor dem ersten PythonEngine.Initialize() Aufruf geschehen
            if (!_isInitialized && _pythonHome != null)
            {
                lock (_lockObject)
                {
                    if (!_isInitialized)
                    {
                        // Finde die tatsächliche Python DLL (nicht python.exe!)
                        // Lese pyvenv.cfg um die Base-Python-Installation zu finden
                        string? basePythonDir = FindBasePythonDirectory(_pythonHome);

                        if (basePythonDir != null)
                        {
                            // Suche nach pythonXY.dll (z.B. python311.dll für Python 3.11)
                            var pythonDlls = Directory.GetFiles(basePythonDir, "python3*.dll");
                            if (pythonDlls.Length > 0)
                            {
                                // Bevorzuge die spezifische Version (python311.dll) über python3.dll
                                var specificDll = pythonDlls.FirstOrDefault(dll => !dll.EndsWith("python3.dll")) ?? pythonDlls[0];
                                _logger.Information("Setting Python DLL path: {PythonDll}", specificDll);
                                Runtime.PythonDLL = specificDll;
                            }
                            else
                            {
                                _logger.Warning("No Python DLL found in {BasePythonDir}", basePythonDir);
                            }
                        }

                        // Setze PYTHONHOME auf venv (nicht Base-Installation!)
                        // Dies stellt sicher, dass Python die venv-Packages findet
                        Environment.SetEnvironmentVariable("PYTHONHOME", _pythonHome);

                        // PYTHONPATH mit venv und Base-Python site-packages
                        var pythonPath = Path.Combine(_pythonHome, "Lib", "site-packages");
                        if (basePythonDir != null)
                        {
                            pythonPath += ";" + Path.Combine(basePythonDir, "Lib", "site-packages");
                        }
                        Environment.SetEnvironmentVariable("PYTHONPATH", pythonPath);

                        _logger.Information("PYTHONHOME set to: {PythonHome}", _pythonHome);
                        _logger.Information("PYTHONPATH set to: {PythonPath}", pythonPath);
                    }
                }
            }
        }

        /// <summary>
        /// Findet das Base-Python-Verzeichnis aus pyvenv.cfg
        /// </summary>
        private static string? FindBasePythonDirectory(string venvPath)
        {
            try
            {
                var pyvenvCfg = Path.Combine(venvPath, "pyvenv.cfg");
                if (!File.Exists(pyvenvCfg))
                    return null;

                var lines = File.ReadAllLines(pyvenvCfg);
                foreach (var line in lines)
                {
                    if (line.StartsWith("home = "))
                    {
                        return line.Substring("home = ".Length).Trim();
                    }
                }
            }
            catch
            {
                // Ignoriere Fehler beim Lesen der Konfiguration
            }

            return null;
        }

        /// <summary>
        /// Initialisiert Python.NET Engine
        /// </summary>
        private void InitializePython()
        {
            if (_isInitialized)
                return;

            lock (_lockObject)
            {
                if (_isInitialized)
                    return;

                try
                {
                    _logger.Information("Initializing Python.NET engine");

                    // Initialisiere Python Engine
                    // Runtime.PythonDLL muss bereits im Constructor gesetzt worden sein
                    PythonEngine.Initialize();
                    PythonEngine.BeginAllowThreads();

                    _isInitialized = true;
                    _logger.Information("Python.NET engine initialized successfully");
                }
                catch (Exception ex)
                {
                    _logger.Error(ex, "Failed to initialize Python.NET engine");
                    throw new InvalidOperationException("Failed to initialize Python engine", ex);
                }
            }
        }

        /// <summary>
        /// Analysiert Kammern für eine Bauteil-JSON-Datei
        /// </summary>
        public ChamberAnalysisResult AnalyzeChambers(string jsonFilePath, string outputDir = null)
        {
            _logger.Information("Analyzing chambers for file: {FilePath}", jsonFilePath);

            if (!File.Exists(jsonFilePath))
            {
                _logger.Error("JSON file not found: {FilePath}", jsonFilePath);
                throw new FileNotFoundException($"JSON file not found: {jsonFilePath}");
            }

            if (!File.Exists(_pythonScriptPath))
            {
                _logger.Error("Python script not found: {ScriptPath}", _pythonScriptPath);
                throw new FileNotFoundException($"Python script not found: {_pythonScriptPath}");
            }

            InitializePython();

            try
            {
                using (Py.GIL())
                {
                    _logger.Debug("Acquiring Python GIL");

                    // Füge Skript-Verzeichnis und venv site-packages zum Python-Path hinzu
                    dynamic sys = Py.Import("sys");

                    // Umleite Python stdout/stderr zu StringIO für Logging
                    dynamic io = Py.Import("io");
                    dynamic stdout = io.StringIO();
                    dynamic stderr = io.StringIO();
                    sys.stdout = stdout;
                    sys.stderr = stderr;
                    string? scriptDir = Path.GetDirectoryName(_pythonScriptPath);

                    // Stelle sicher, dass venv site-packages im Path ist
                    if (_pythonHome != null)
                    {
                        var venvSitePackages = Path.Combine(_pythonHome, "Lib", "site-packages");
                        if (Directory.Exists(venvSitePackages))
                        {
                            using (PyObject pyVenvSitePackages = new PyString(venvSitePackages))
                            {
                                if (!sys.path.__contains__(pyVenvSitePackages))
                                {
                                    _logger.Debug("Adding venv site-packages to sys.path: {Path}", venvSitePackages);
                                    sys.path.insert(0, venvSitePackages);
                                }
                            }
                        }
                    }

                    // Füge Skript-Verzeichnis hinzu
                    if (scriptDir != null)
                    {
                        // Verwende Python's own __contains__ Methode statt C# cast
                        using (PyObject pyScriptDir = new PyString(scriptDir))
                        {
                            if (!sys.path.__contains__(pyScriptDir))
                            {
                                _logger.Debug("Adding script directory to sys.path: {Path}", scriptDir);
                                sys.path.insert(0, scriptDir);
                            }
                        }
                    }

                    // Importiere das Python-Modul
                    _logger.Debug("Importing Python module");
                    dynamic chamberAnalyzer = Py.Import("chamber_analyzer_json");

                    // Rufe Analyse-Funktion auf
                    _logger.Debug("Calling analyze_chambers_from_json");
                    dynamic result = chamberAnalyzer.analyze_chambers_from_json(
                        jsonFilePath,
                        outputDir ?? (object?)null
                    );

                    _logger.Debug("Python function returned, checking result");

                    // Lese Python stdout/stderr
                    stdout.seek(0);
                    stderr.seek(0);
                    string pythonStdout = stdout.read().ToString();
                    string pythonStderr = stderr.read().ToString();

                    if (!string.IsNullOrWhiteSpace(pythonStdout))
                    {
                        _logger.Information("Python stdout: {Stdout}", pythonStdout);
                    }
                    if (!string.IsNullOrWhiteSpace(pythonStderr))
                    {
                        _logger.Error("Python stderr: {Stderr}", pythonStderr);
                    }

                    // Prüfe ob result None ist
                    if (result == null || result is Python.Runtime.PyObject && result.IsNone())
                    {
                        _logger.Error("Python function returned None");
                        return new ChamberAnalysisResult
                        {
                            Success = false,
                            ErrorMessage = "Python function returned None - check Python logs for errors",
                            ChamberCenters = new List<ChamberCenter>()
                        };
                    }

                    // Konvertiere Python-Ergebnis zu C#
                    bool success = (bool)result["success"];

                    if (!success)
                    {
                        string error = "Unknown error";
                        try
                        {
                            if (result.__contains__("error"))
                            {
                                error = result["error"]?.ToString() ?? "Unknown error";
                            }
                        }
                        catch
                        {
                            // Fallback to Unknown error
                        }

                        _logger.Error("Chamber analysis failed: {Error}", error);

                        return new ChamberAnalysisResult
                        {
                            Success = false,
                            ErrorMessage = error,
                            ChamberCenters = new List<ChamberCenter>()
                        };
                    }

                    // Extrahiere Chamber Centers
                    var chamberCenters = new List<ChamberCenter>();
                    dynamic centersArray = result["chamber_centers"];

                    foreach (dynamic centerData in centersArray)
                    {
                        var chamberCenter = new ChamberCenter
                        {
                            ConnectionPointIndex = (int)centerData["connection_point_index"],
                            ConnectionPointName = centerData["connection_point_name"].ToString()
                        };

                        if (centerData["chamber_center"] != null)
                        {
                            dynamic center = centerData["chamber_center"];
                            chamberCenter.Center = new Point3DData
                            {
                                X = (double)center["X"],
                                Y = (double)center["Y"],
                                Z = (double)center["Z"]
                            };
                        }

                        chamberCenters.Add(chamberCenter);
                    }

                    int contourCount = 0;
                    try
                    {
                        if (result.__contains__("contour_count"))
                        {
                            contourCount = (int)result["contour_count"];
                        }
                    }
                    catch
                    {
                        // Fallback to 0
                    }

                    _logger.Information("Chamber analysis successful: {ChamberCount} centers found, {ContourCount} contours",
                        chamberCenters.Count, contourCount);

                    return new ChamberAnalysisResult
                    {
                        Success = true,
                        ChamberCenters = chamberCenters,
                        ContourCount = contourCount
                    };
                }
            }
            catch (PythonException pyEx)
            {
                _logger.Error(pyEx, "Python exception during chamber analysis");
                return new ChamberAnalysisResult
                {
                    Success = false,
                    ErrorMessage = $"Python error: {pyEx.Message}",
                    ChamberCenters = new List<ChamberCenter>()
                };
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Unexpected error during chamber analysis");
                return new ChamberAnalysisResult
                {
                    Success = false,
                    ErrorMessage = $"Unexpected error: {ex.Message}",
                    ChamberCenters = new List<ChamberCenter>()
                };
            }
        }

        public void Dispose()
        {
            if (_isInitialized)
            {
                try
                {
                    _logger.Information("Shutting down Python.NET engine");
                    PythonEngine.Shutdown();
                    _isInitialized = false;
                }
                catch (Exception ex)
                {
                    _logger.Error(ex, "Error shutting down Python.NET engine");
                }
            }
        }
    }

    /// <summary>
    /// Ergebnis der Kammer-Analyse
    /// </summary>
    public class ChamberAnalysisResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = string.Empty;
        public List<ChamberCenter> ChamberCenters { get; set; } = new();
        public int ContourCount { get; set; }
    }

    /// <summary>
    /// Kammer-Mittelpunkt für einen Connection Point
    /// </summary>
    public class ChamberCenter
    {
        public int ConnectionPointIndex { get; set; }
        public string ConnectionPointName { get; set; } = string.Empty;
        public Point3DData? Center { get; set; }
    }
}
