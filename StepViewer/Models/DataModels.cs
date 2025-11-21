using System.Collections.Generic;
using Newtonsoft.Json;

namespace StepViewer.Models
{
    /// <summary>
    /// Root model for DataSet JSON files
    /// </summary>
    public class PartData
    {
        [JsonProperty("PartNr")]
        public string PartNr { get; set; } = string.Empty;

        [JsonProperty("Graphic3d")]
        public Graphic3D Graphic3d { get; set; } = new();

        [JsonProperty("BoundingBox")]
        public BoundingBox BoundingBox { get; set; } = new();

        [JsonProperty("ConnectionPoints")]
        public List<ConnectionPoint> ConnectionPoints { get; set; } = new();

        [JsonProperty("ProductTopGroup")]
        public int ProductTopGroup { get; set; }

        [JsonProperty("ProductGroup")]
        public int ProductGroup { get; set; }

        [JsonProperty("ProductSubGroup")]
        public int ProductSubGroup { get; set; }
    }

    /// <summary>
    /// 3D Graphics data containing mesh points and triangle indices
    /// </summary>
    public class Graphic3D
    {
        [JsonProperty("Points")]
        public List<Point3DData> Points { get; set; } = new();

        [JsonProperty("Indices")]
        public List<int> Indices { get; set; } = new();
    }

    /// <summary>
    /// 3D Point with X, Y, Z coordinates
    /// </summary>
    public class Point3DData
    {
        [JsonProperty("X")]
        public double X { get; set; }

        [JsonProperty("Y")]
        public double Y { get; set; }

        [JsonProperty("Z")]
        public double Z { get; set; }

        public Point3DData() { }

        public Point3DData(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public override string ToString()
        {
            return $"({X:F3}, {Y:F3}, {Z:F3})";
        }
    }

    /// <summary>
    /// Bounding box with dimension and location
    /// </summary>
    public class BoundingBox
    {
        [JsonProperty("Dimension")]
        public Point3DData Dimension { get; set; } = new();

        [JsonProperty("Location")]
        public Point3DData Location { get; set; } = new();
    }

    /// <summary>
    /// Connection point with position and insertion direction
    /// </summary>
    public class ConnectionPoint
    {
        [JsonProperty("Index")]
        public int Index { get; set; }

        [JsonProperty("Name")]
        public string Name { get; set; } = string.Empty;

        [JsonProperty("Point")]
        public Point3DData Point { get; set; } = new();

        [JsonProperty("InsertDirection")]
        public Point3DData InsertDirection { get; set; } = new();

        public override string ToString()
        {
            return $"{Index}: {Name} @ {Point}";
        }
    }
}
