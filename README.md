# 🧬 NicheFinder — Spatial Biology Explorer

> **Tagline:** *Discover the hidden neighbourhoods of cells.*  
> **Interactive Spatial Biology Prototype** · College Hackathon Edition

---

## 📌 Project Overview

**NicheFinder** is an interactive spatial biology web application designed for exploring single-cell spatial distributions in biological tissue samples. It maps cellular coordinates, reveals tissue microenvironments, and computes Euclidean distances between cells.

---

## 🚀 Supported Structured Input Formats

NicheFinder supports 5 structured data formats with automatic parsing:
- **CSV** (`.csv`): Standard comma-separated tables.
- **TSV** (`.tsv`, `.tab`): Tab-separated tabular data.
- **Excel** (`.xlsx`, `.xls`): Workbooks read via `openpyxl`.
- **JSON** (`.json`): Array of cell records or column-oriented dictionaries.
- **Parquet** (`.parquet`, `.pq`): High-performance columnar binary format via `pyarrow`.

> **Note on Tissue Images:** Microscopy images (`.png`, `.jpg`, `.tif`, `.tiff`) contain raw pixel matrices rather than structured cell coordinates. NicheFinder automatically detects uploaded images and provides guidance explaining that structured coordinate tables are required.

### Required Fields:
- `cell_type`: Biological phenotype or identity annotation (taken directly from data; not guessed from coordinates).
- `x`: Horizontal spatial coordinate (numeric).
- `y`: Vertical spatial coordinate (numeric).
- `cell_id` *(optional)*: Unique identifier for each cell (generated if absent).

---

## 📊 Dashboard Structure (Sequential Single-Page Flow)

1. **SAMPLE OVERVIEW (Top Metric Cards)**:
   - Total Cells, Number of Annotated Cell Types, Most Abundant Cell Type.
2. **Cell Types Present**:
   - Tabular breakdown of Cell Type, Number of Cells, and Percentage of Total Cells.
   - Includes instant CSV download.
3. **Cell Counts and Percentages**:
   - Interactive Bar Chart of cell counts by type.
   - Interactive Pie/Donut Chart of cell-type proportions.
4. **Spatial Arrangement**:
   - WebGL-accelerated 2D scatter map with 1:1 aspect ratio.
   - Distinct color per cell type, clear legend, zoom, pan, and sub-cellular hover tooltips.
   - Star marker for selected focal cell, cyan markers for neighbours, dashed radius circle, and connecting distance lines.
5. **Explore Neighbouring Cells**:
   - Focal cell selection dropdown + interactive neighbourhood radius slider.
   - Distance-sorted table of neighbours: Neighbour ID, Cell Type, X, Y, and Euclidean Distance.
   - Downloadable CSV table.
6. **Distance Between Cells**:
   - Bar chart of actual distances from the selected cell to each neighbouring cell in **coordinate units** (ordered from nearest to farthest).
   - Distances are clearly labeled in coordinate units, never as a percentage.
7. **Neighbour Composition**:
   - Table showing cell type counts and percentages within the neighbourhood (summing to exactly 100.0%).
   - Horizontal bar chart and pie chart of local neighbour proportions.

---

## 💻 Exact Windows Commands to Run the Application

### 1. Open Windows PowerShell in your project directory:
```powershell
cd "c:\Users\Mounika Arya\OneDrive\Documents\NicheFinder"
```

### 2. (Optional) Run the automated test suite:
```powershell
.\.venv\Scripts\python.exe test_app.py
```
*(All 7 core test suites will report `[PASS]`)*

### 3. Launch the Streamlit application:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

*The application will open automatically in your browser at **`http://localhost:8501`**.*

---

## 🔑 Demonstration Authentication

- **Lead Bioinformatician**: `researcher@biohack.org` / `spatial2026`
- **Evaluator**: `demo` / `nichefinder`
- **One-Click Express Login**: Click **"⚡ Express Demo Login"** on the welcome screen.
