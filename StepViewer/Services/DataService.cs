using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Serilog;
using StepViewer.Models;

namespace StepViewer.Services
{
    /// <summary>
    /// Service for loading and managing part data from JSON files
    /// </summary>
    public class DataService
    {
        private readonly string _dataSetPath;
        private readonly ILogger _logger;

        public DataService(string? dataSetPath = null)
        {
            _logger = Log.ForContext<DataService>();

            // Default to DataSet folder in parent directory of application
            _dataSetPath = dataSetPath ?? Path.Combine(
                Directory.GetParent(AppDomain.CurrentDomain.BaseDirectory)?.Parent?.Parent?.Parent?.Parent?.FullName ?? "",
                "DataSet"
            );

            _logger.Information("DataService initialized with path: {DataSetPath}", _dataSetPath);
        }

        /// <summary>
        /// Get all JSON file paths in the DataSet directory
        /// </summary>
        public List<string> GetAllJsonFiles()
        {
            _logger.Debug("Searching for JSON files in: {DataSetPath}", _dataSetPath);

            if (!Directory.Exists(_dataSetPath))
            {
                _logger.Error("DataSet directory not found: {DataSetPath}", _dataSetPath);
                throw new DirectoryNotFoundException($"DataSet directory not found: {_dataSetPath}");
            }

            var files = Directory.GetFiles(_dataSetPath, "*.json", SearchOption.TopDirectoryOnly)
                .OrderBy(f => Path.GetFileName(f))
                .ToList();

            _logger.Information("Found {FileCount} JSON files in DataSet directory", files.Count);
            return files;
        }

        /// <summary>
        /// Load part data from a JSON file
        /// </summary>
        public PartData LoadPartData(string filePath)
        {
            _logger.Debug("Loading part data from: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                _logger.Error("File not found: {FilePath}", filePath);
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            try
            {
                string jsonContent = File.ReadAllText(filePath);
                _logger.Debug("JSON content length: {Length} characters", jsonContent.Length);

                var partData = JsonConvert.DeserializeObject<PartData>(jsonContent);

                if (partData == null)
                {
                    _logger.Error("Failed to deserialize JSON from: {FilePath}", filePath);
                    throw new InvalidDataException($"Failed to deserialize JSON from: {filePath}");
                }

                _logger.Information("Successfully loaded part data: PartNr={PartNr}, Points={PointCount}, Connections={ConnectionCount}",
                    partData.PartNr,
                    partData.Graphic3d?.Points?.Count ?? 0,
                    partData.ConnectionPoints?.Count ?? 0);

                return partData;
            }
            catch (JsonException ex)
            {
                _logger.Error(ex, "Invalid JSON format in file: {FilePath}", filePath);
                throw new InvalidDataException($"Invalid JSON format in file: {filePath}", ex);
            }
        }

        /// <summary>
        /// Save part data to a JSON file
        /// </summary>
        public void SavePartData(string filePath, PartData partData)
        {
            _logger.Information("Saving part data to: {FilePath}", filePath);

            try
            {
                string jsonContent = JsonConvert.SerializeObject(partData, Formatting.Indented);
                File.WriteAllText(filePath, jsonContent);
                _logger.Information("Successfully saved part data to: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Failed to save JSON to file: {FilePath}", filePath);
                throw new IOException($"Failed to save JSON to file: {filePath}", ex);
            }
        }

        /// <summary>
        /// Get file info with part number and connection count
        /// </summary>
        public (string fileName, string partNr, int connectionCount) GetFileInfo(string filePath)
        {
            try
            {
                var partData = LoadPartData(filePath);
                return (
                    Path.GetFileName(filePath),
                    partData.PartNr,
                    partData.ConnectionPoints?.Count ?? 0
                );
            }
            catch (Exception ex)
            {
                _logger.Warning(ex, "Failed to get file info for: {FilePath}", filePath);
                return (Path.GetFileName(filePath), "Error", 0);
            }
        }
    }
}
