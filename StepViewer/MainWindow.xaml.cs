using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using HelixToolkit.Wpf;
using Serilog;
using StepViewer.Models;
using StepViewer.Services;

namespace StepViewer
{
    /// <summary>
    /// Main Window for 3D Part Viewer
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly DataService _dataService;
        private PartData? _currentPartData;
        private readonly List<Visual3D> _meshVisuals = new List<Visual3D>();
        private readonly List<Visual3D> _connectionVisuals = new List<Visual3D>();
        private readonly ILogger _logger;

        public MainWindow()
        {
            InitializeComponent();

            _logger = Log.ForContext<MainWindow>();
            _logger.Information("MainWindow initializing");

            // Initialize data service
            _dataService = new DataService();

            // Load file list
            LoadFileList();

            _logger.Information("MainWindow initialized successfully");
        }

        /// <summary>
        /// Load all JSON files into the file list
        /// </summary>
        private void LoadFileList()
        {
            try
            {
                _logger.Debug("Loading file list from DataSet directory");
                var jsonFiles = _dataService.GetAllJsonFiles();
                var fileInfoList = new List<FileInfo>();

                foreach (var filePath in jsonFiles)
                {
                    var (fileName, partNr, connectionCount) = _dataService.GetFileInfo(filePath);
                    fileInfoList.Add(new FileInfo
                    {
                        FilePath = filePath,
                        FileName = fileName,
                        PartNr = partNr,
                        ConnectionCount = connectionCount
                    });
                }

                FileListBox.ItemsSource = fileInfoList;
                StatusText.Text = $"{fileInfoList.Count} Bauteile geladen";
                _logger.Information("Loaded {FileCount} files from DataSet directory", fileInfoList.Count);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to load file list");
                MessageBox.Show($"Fehler beim Laden der Dateiliste:\n{ex.Message}",
                    "Fehler", MessageBoxButton.OK, MessageBoxImage.Error);
                StatusText.Text = "Fehler beim Laden der Dateiliste";
            }
        }

        /// <summary>
        /// File selection changed event
        /// </summary>
        private void FileListBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (FileListBox.SelectedItem is FileInfo fileInfo)
            {
                LoadPartData(fileInfo.FilePath);
            }
        }

        /// <summary>
        /// Load part data from file and display in 3D viewport
        /// </summary>
        private void LoadPartData(string filePath)
        {
            try
            {
                _logger.Information("Loading part data from file: {FilePath}", filePath);
                StatusText.Text = $"Lade {Path.GetFileName(filePath)}...";

                // Load part data
                _currentPartData = _dataService.LoadPartData(filePath);
                _logger.Debug("Part data loaded: PartNr={PartNr}, Points={PointCount}, Indices={IndexCount}, Connections={ConnectionCount}",
                    _currentPartData.PartNr,
                    _currentPartData.Graphic3d.Points.Count,
                    _currentPartData.Graphic3d.Indices.Count,
                    _currentPartData.ConnectionPoints.Count);

                // Clear existing 3D objects
                ClearScene();

                // Render mesh
                if (ShowMeshCheckBox.IsChecked == true)
                {
                    _logger.Debug("Rendering mesh");
                    RenderMesh();
                }

                // Render connection points
                if (ShowConnectionsCheckBox.IsChecked == true)
                {
                    _logger.Debug("Rendering {ConnectionCount} connection points", _currentPartData.ConnectionPoints.Count);
                    RenderConnectionPoints();
                }

                // Update connection points list
                ConnectionPointsList.ItemsSource = _currentPartData.ConnectionPoints;

                // Zoom to fit
                Viewport3D.ZoomExtents();

                var statusMessage = $"{Path.GetFileName(filePath)} geladen - " +
                    $"{_currentPartData.Graphic3d.Points.Count} Punkte, " +
                    $"{_currentPartData.Graphic3d.Indices.Count / 3} Dreiecke, " +
                    $"{_currentPartData.ConnectionPoints.Count} Anschlüsse";
                StatusText.Text = statusMessage;
                _logger.Information("Part data loaded and rendered successfully: {StatusMessage}", statusMessage);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to load part data from file: {FilePath}", filePath);
                MessageBox.Show($"Fehler beim Laden der Datei:\n{ex.Message}",
                    "Fehler", MessageBoxButton.OK, MessageBoxImage.Error);
                StatusText.Text = "Fehler beim Laden der Datei";
            }
        }

        /// <summary>
        /// Clear all 3D objects from scene
        /// </summary>
        private void ClearScene()
        {
            if (Viewport3D == null) return;

            foreach (var visual in _meshVisuals)
            {
                Viewport3D.Children.Remove(visual);
            }
            _meshVisuals.Clear();

            foreach (var visual in _connectionVisuals)
            {
                Viewport3D.Children.Remove(visual);
            }
            _connectionVisuals.Clear();
        }

        /// <summary>
        /// Render the 3D mesh from Points and Indices
        /// </summary>
        private void RenderMesh()
        {
            if (_currentPartData?.Graphic3d == null)
            {
                _logger.Warning("Cannot render mesh: No part data loaded");
                return;
            }

            _logger.Debug("Starting mesh rendering");
            var points = _currentPartData.Graphic3d.Points;
            var indices = _currentPartData.Graphic3d.Indices;

            // Convert Points to Point3D collection
            var point3DCollection = new Point3DCollection();
            foreach (var pt in points)
            {
                point3DCollection.Add(new System.Windows.Media.Media3D.Point3D(pt.X, pt.Y, pt.Z));
            }

            // Convert Indices to Int32Collection
            var indexCollection = new Int32Collection();
            foreach (var idx in indices)
            {
                indexCollection.Add(idx);
            }

            // Create mesh geometry
            var meshGeometry = new MeshGeometry3D
            {
                Positions = point3DCollection,
                TriangleIndices = indexCollection
            };

            // Calculate normals for proper lighting
            meshGeometry.Normals = CalculateNormals(meshGeometry);

            // Create material (gray with some shine)
            var material = new DiffuseMaterial(new SolidColorBrush(Color.FromRgb(180, 180, 180)));
            var specularMaterial = new SpecularMaterial(new SolidColorBrush(Colors.White), 30);
            var materialGroup = new MaterialGroup();
            materialGroup.Children.Add(material);
            materialGroup.Children.Add(specularMaterial);

            // Create GeometryModel3D
            var geometryModel = new GeometryModel3D
            {
                Geometry = meshGeometry,
                Material = materialGroup,
                BackMaterial = materialGroup
            };

            // Create ModelVisual3D
            var modelVisual = new ModelVisual3D
            {
                Content = geometryModel
            };

            // Add to viewport
            Viewport3D.Children.Add(modelVisual);
            _meshVisuals.Add(modelVisual);
            _logger.Debug("Mesh rendered successfully with {TriangleCount} triangles", indices.Count / 3);
        }

        /// <summary>
        /// Calculate normals for mesh geometry
        /// </summary>
        private Vector3DCollection CalculateNormals(MeshGeometry3D mesh)
        {
            var normals = new Vector3DCollection(mesh.Positions.Count);
            for (int i = 0; i < mesh.Positions.Count; i++)
            {
                normals.Add(new Vector3D(0, 0, 0));
            }

            // Calculate face normals and accumulate
            for (int i = 0; i < mesh.TriangleIndices.Count; i += 3)
            {
                int idx0 = mesh.TriangleIndices[i];
                int idx1 = mesh.TriangleIndices[i + 1];
                int idx2 = mesh.TriangleIndices[i + 2];

                var p0 = mesh.Positions[idx0];
                var p1 = mesh.Positions[idx1];
                var p2 = mesh.Positions[idx2];

                var v1 = p1 - p0;
                var v2 = p2 - p0;
                var normal = Vector3D.CrossProduct(v1, v2);
                normal.Normalize();

                normals[idx0] += normal;
                normals[idx1] += normal;
                normals[idx2] += normal;
            }

            // Normalize all normals
            for (int i = 0; i < normals.Count; i++)
            {
                var n = normals[i];
                n.Normalize();
                normals[i] = n;
            }

            return normals;
        }

        /// <summary>
        /// Render connection points as spheres with arrows
        /// </summary>
        private void RenderConnectionPoints()
        {
            if (_currentPartData?.ConnectionPoints == null)
                return;

            foreach (var connectionPoint in _currentPartData.ConnectionPoints)
            {
                // Create sphere at connection point
                var sphere = new SphereVisual3D
                {
                    Center = new System.Windows.Media.Media3D.Point3D(
                        connectionPoint.Point.X,
                        connectionPoint.Point.Y,
                        connectionPoint.Point.Z),
                    Radius = CalculatePointRadius(),
                    Fill = new SolidColorBrush(Colors.Red)
                };

                Viewport3D.Children.Add(sphere);
                _connectionVisuals.Add(sphere);

                // Create arrow for insert direction
                var arrowLength = CalculateArrowLength();
                var endPoint = new System.Windows.Media.Media3D.Point3D(
                    connectionPoint.Point.X + connectionPoint.InsertDirection.X * arrowLength,
                    connectionPoint.Point.Y + connectionPoint.InsertDirection.Y * arrowLength,
                    connectionPoint.Point.Z + connectionPoint.InsertDirection.Z * arrowLength);

                var arrow = new ArrowVisual3D
                {
                    Point1 = new System.Windows.Media.Media3D.Point3D(
                        connectionPoint.Point.X,
                        connectionPoint.Point.Y,
                        connectionPoint.Point.Z),
                    Point2 = endPoint,
                    Diameter = CalculatePointRadius() * 0.5,
                    Fill = new SolidColorBrush(Colors.Blue)
                };

                Viewport3D.Children.Add(arrow);
                _connectionVisuals.Add(arrow);

                // Add label (text billboard)
                var textVisual = new BillboardTextVisual3D
                {
                    Text = $"{connectionPoint.Index}: {connectionPoint.Name}",
                    Position = new System.Windows.Media.Media3D.Point3D(
                        connectionPoint.Point.X,
                        connectionPoint.Point.Y,
                        connectionPoint.Point.Z + CalculatePointRadius() * 2),
                    Foreground = new SolidColorBrush(Colors.Black),
                    FontSize = 12
                };

                Viewport3D.Children.Add(textVisual);
                _connectionVisuals.Add(textVisual);
            }
        }

        /// <summary>
        /// Calculate appropriate point radius based on model size
        /// </summary>
        private double CalculatePointRadius()
        {
            if (_currentPartData?.BoundingBox?.Dimension == null)
                return 1.0;

            var dimension = _currentPartData.BoundingBox.Dimension;
            var maxDim = Math.Max(Math.Max(dimension.X, dimension.Y), dimension.Z);
            return maxDim * 0.02; // 2% of max dimension
        }

        /// <summary>
        /// Calculate appropriate arrow length based on model size
        /// </summary>
        private double CalculateArrowLength()
        {
            if (_currentPartData?.BoundingBox?.Dimension == null)
                return 10.0;

            var dimension = _currentPartData.BoundingBox.Dimension;
            var maxDim = Math.Max(Math.Max(dimension.X, dimension.Y), dimension.Z);
            return maxDim * 0.15; // 15% of max dimension
        }

        // UI Event Handlers

        private void ShowMeshCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            // Check if viewport is initialized (avoid errors during XAML initialization)
            if (Viewport3D == null || _currentPartData == null) return;

            foreach (var visual in _meshVisuals)
            {
                Viewport3D.Children.Remove(visual);
            }
            _meshVisuals.Clear();

            if (ShowMeshCheckBox.IsChecked == true)
            {
                RenderMesh();
            }
        }

        private void ShowConnectionsCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            // Check if viewport is initialized (avoid errors during XAML initialization)
            if (Viewport3D == null || _currentPartData == null) return;

            foreach (var visual in _connectionVisuals)
            {
                Viewport3D.Children.Remove(visual);
            }
            _connectionVisuals.Clear();

            if (ShowConnectionsCheckBox.IsChecked == true)
            {
                RenderConnectionPoints();
            }
        }

        private void ShowCoordinateSystemCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            // Check if viewport is initialized (avoid errors during XAML initialization)
            if (Viewport3D == null) return;

            Viewport3D.ShowCoordinateSystem = ShowCoordinateSystemCheckBox.IsChecked == true;
        }

        private void ZoomIn_Click(object sender, RoutedEventArgs e)
        {
            if (Viewport3D == null) return;

            _logger.Debug("Zoom in requested");
            Viewport3D.ZoomExtents(0.8);
        }

        private void ZoomOut_Click(object sender, RoutedEventArgs e)
        {
            if (Viewport3D == null) return;

            _logger.Debug("Zoom out requested");
            Viewport3D.ZoomExtents(1.2);
        }

        private void ResetView_Click(object sender, RoutedEventArgs e)
        {
            if (Viewport3D == null) return;

            _logger.Debug("Reset view requested");
            Viewport3D.ZoomExtents();
        }

        /// <summary>
        /// Helper class for file list binding
        /// </summary>
        private class FileInfo
        {
            public string FilePath { get; set; } = string.Empty;
            public string FileName { get; set; } = string.Empty;
            public string PartNr { get; set; } = string.Empty;
            public int ConnectionCount { get; set; }
        }
    }
}