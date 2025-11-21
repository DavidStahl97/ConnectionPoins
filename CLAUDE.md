# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**STEP-Anschlussengineering** is a Python application for automated chamber opening detection in 3D CAD components. It processes STEP (.stp/.step) files to identify connection points and analyze chamber geometries using voxel-based depth image analysis and OpenCV contour detection.

The system consists of two main applications:
1. **step_3d_viewer.py** - Interactive PyQt6-based 3D viewer for manual connection point selection
2. **connection_chamber_analyzer.py** - Automated chamber opening detection and 3D center point calculation

## Setup and Installation

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Applications

### Interactive 3D Viewer
```bash
python step_3d_viewer.py
```
- Loads STEP files from the `DataSet/` directory (displayed in sidebar)
- Click on surfaces to define connection points
- Vectors automatically point in world Z-direction (0, 0, 1)
- Saves connection vectors to `{filename}.json` in the same directory as the STEP file
- Integrates with chamber analyzer (button in UI)

### Chamber Analyzer
```bash
# Batch mode - process all STEP files in DataSet/ with existing JSON files
python connection_chamber_analyzer.py

# Single file mode
python connection_chamber_analyzer.py filename.stp
python connection_chamber_analyzer.py DataSet/filename.stp
```
- Automatically detects chamber openings using voxel-based depth projection (800x resolution)
- Calculates 3D chamber center points with maximum depth (Z-coordinate)
- Updates JSON files with `chamber_center` data for each connection vector
- Generates analysis images: depth maps, gradients, contour detection pipeline

## Key Architecture Details

### Coordinate System
- Uses world coordinate system where Z-axis points "out of the screen" toward the user
- All connection vectors in the viewer point in positive Z direction: `[0.0, 0.0, 1.0]`
- Mesh is normalized to unit size centered at origin for consistent rendering
- Original mesh transformations stored as `mesh_center` and `mesh_scale` for coordinate mapping

### Data Flow (step_3d_viewer.py → connection_chamber_analyzer.py)
1. User loads STEP file in viewer, clicks connection points on 3D surface
2. Viewer saves connection vectors to JSON: `{filename}.json`
3. Analyzer loads STEP file + JSON, performs voxel-based chamber detection
4. Analyzer updates JSON with `chamber_center` coordinates (X, Y, Z)
5. Viewer can reload JSON to visualize both connection points (white) and chamber centers (yellow)

### JSON Data Structure
```json
{
  "source_step_file": "filename.stp",
  "connection_vectors": [
    {
      "id": 1,
      "position": {"x": 0.002147, "y": 0.043655, "z": 0.017129},
      "direction": {"x": 0.0, "y": 0.0, "z": 1.0},
      "chamber_center": {"x": 0.002822, "y": 0.042836, "z": 0.035235}
    }
  ]
}
```
- `position`: User-selected connection point (surface click in viewer)
- `direction`: Always `[0, 0, 1]` in world coordinates
- `chamber_center`: Automatically detected by analyzer (added later, may be null)

### Chamber Detection Algorithm (connection_chamber_analyzer.py)
1. **Voxelization**: Convert STEP mesh to voxel grid (resolution=800 for precision)
2. **Depth Projection**: Create Z-direction depth image (top-down view)
3. **Gradient Analysis**: Apply Sobel operators to detect edges in depth map
4. **Contour Detection**: Use OpenCV to find closed contours in gradient magnitude
5. **Edge Completion**: Add frame border to close contours cut off at image edges
6. **Hierarchy Filtering**: Remove nested contours (inner holes) to keep only outer chambers
7. **Connection Matching**: Associate each connection point with its containing contour
8. **Center Calculation**: Compute bounding box center + maximum depth (Z) from contour edge pixels

### OpenGL Rendering Optimizations (step_3d_viewer.py)
- **Display Lists**: Pre-compiled mesh geometry for fast rendering (`glCallList`)
- **Vertex Arrays**: Batched vertex/normal data to minimize draw calls
- **Back-face Culling**: Enabled to skip non-visible faces
- **Bug Fix**: Display list is explicitly deleted before loading new mesh (`glDeleteLists`) to prevent rendering artifacts

### File Organization
- `DataSet/` - Contains STEP files (.stp/.step) and corresponding JSON files
- `DataSet/{filename}/` - Auto-created output directory for each analyzed file
  - `{filename}_depth_image.png` - High-resolution depth map (800x resolution)
  - `{filename}_gradient_magnitude.png` - Edge detection visualization
  - `{filename}_contours_analysis.png` - 4-panel pipeline visualization showing:
    - Panel 1: Original binary image with connection points
    - Panel 2: All completed contours (includes edge completions)
    - Panel 3: Filtered contours (nested/inner contours removed)
    - Panel 4: Final chamber contours with connection points and centers
  - `{filename}_contours_filtered.png` - Final detected chambers only

### Vector Count Display in STEP File List
The viewer's file list shows `(X/Y) filename.stp` format where:
- X = number of vectors with chamber centers detected
- Y = total number of connection vectors defined
- Indicates analysis completion status at a glance

## Critical Implementation Notes

### OpenGL Context and Ray-Casting
- Uses `glReadPixels` with depth buffer for accurate 3D point selection
- Requires `makeCurrent()` before depth buffer reads
- `gluUnProject` transforms screen coordinates to world space using OpenGL matrices
- Points are transformed back to original mesh coordinates using stored `mesh_scale` and `mesh_center`

### Mesh Loading with trimesh
- STEP files may return either `Scene` (with `.dump()`) or direct `TriangleMesh` objects
- Always check for `.dump()` method first to handle both cases
- First mesh in scene is used if multiple meshes exist

### Chamber Center Z-Coordinate Extraction
- Only samples depth values from **contour edge pixels** (not interior)
- Uses maximum depth value as chamber center Z (deepest/farthest point)
- Avoids bias from filled interior regions which may have varying depths
- Temporary mask with `cv2.drawContours(thickness=1)` isolates edge pixels

### Contour Edge Completion Strategy
- Adds 5-pixel frame border around binary image
- Groups edge points by parent contour to avoid cross-contamination
- Connects only first and last edge point of **same contour** per image side
- Prevents merging of adjacent separate chambers that both touch the edge

### Coordinate System Transformations
Chain of coordinate transforms:
1. **Original Mesh** (from STEP file) → stored in `original_vertices`
2. **Normalized Mesh** (centered at origin, scaled to unit size) → used for rendering
3. **Screen Space** (pixel coordinates) → used for mouse input
4. **Depth Image Space** (voxel projection) → used for chamber detection

Transform parameters stored: `mesh_center`, `mesh_scale`, voxel `extent` [x_min, x_max, y_min, y_max]

## Testing and Validation

No automated tests are present. Manual validation workflow:
1. Load STEP file in viewer and define connection points
2. Run chamber analyzer on same file
3. Reload in viewer to see chamber centers (yellow spheres)
4. Verify chamber centers are geometrically correct in 3D view
5. Check analysis images for correct contour detection

## Dependencies

Core libraries (see requirements.txt):
- PyQt6 - GUI framework for 3D viewer
- PyOpenGL - OpenGL bindings for 3D rendering
- trimesh - STEP file loading and mesh processing
- open3d - Voxelization and 3D geometry operations
- opencv-python - Image processing and contour detection
- numpy - Numerical operations
- matplotlib - Visualization and image export
- scipy - Interpolation for depth image gap filling

## Common Development Tasks

**Adding new connection point features:**
- Modify `on_point_selected()` in STEP3DMainWindow class
- Update JSON schema in `save_points()` method
- Ensure `load_json_vectors()` in analyzer handles new fields

**Changing chamber detection parameters:**
- Voxel resolution: `voxel_resolution=800` in `mesh_to_voxels()`
- Gradient threshold: `threshold_factor=0.1` in `detect_contours()`
- Edge completion frame: `frame_width=5` in `complete_individual_cut_contours()`
- Minimum contour size: `min_contour_area=20` in `detect_contours()`

**Modifying visualization outputs:**
- All matplotlib figures use `figsize` and `dpi=150` for consistent quality
- Update `save_contour_analysis()` to add/remove panels
- Visualization colors defined in `colors` list (8 standard colors for contours)

**Debugging chamber detection failures:**
1. Check `{filename}_depth_image.png` - verify voxel projection quality
2. Check `{filename}_gradient_magnitude.png` - verify edge detection
3. Check `{filename}_contours_analysis.png` Panel 2 - verify edge completion logic
4. Adjust `threshold_factor` if contours are over/under-detected

## Known Issues and Limitations

- Only Z-direction projection is implemented (top-down view)
- Nested contours are always removed - may incorrectly filter complex geometries with intentional holes
- Edge completion assumes cut contours should connect linearly - may not work for curved edges
- No multi-threading for batch processing (processes files sequentially)
- Large STEP files (>100k faces) may cause slow viewer performance despite display lists
