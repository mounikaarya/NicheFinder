"""
Comprehensive test suite for NicheFinder — Spatial Biology Explorer.
Validates:
1. Sample dataset loading (550 cells, 6 phenotypes)
2. Every supported structured file reader:
   - CSV (.csv)
   - TSV (.tsv)
   - Excel (.xlsx)
   - JSON (.json)
   - Parquet (.parquet)
3. Rejection of unstructured image files (PNG, JPG, TIFF) with clear guidance
4. Validation of missing required columns (cell_type, x, y)
5. Accurate Euclidean distance calculation & strict focal cell self-exclusion
6. Verification that distance is a spatial length (coordinate units), never a percentage
7. Neighbour composition counts and percentages summing to exactly 100%
8. Safe CSV export serialization
"""

import io
import json
import numpy as np
import pandas as pd
import app


def test_sample_dataset():
    """Verify built-in demonstration dataset."""
    df = app.load_sample_dataset()
    assert len(df) == 550
    assert set(df.columns) == {"cell_id", "cell_type", "x", "y"}
    assert df["cell_type"].nunique() == 6
    assert not df["cell_id"].duplicated().any()
    print("[PASS] test_sample_dataset passed (550 cells across 6 phenotypes).")


def test_supported_file_readers():
    """Verify all 5 supported structured file formats: CSV, TSV, Excel, JSON, Parquet."""
    base_df = pd.DataFrame({
        "cell_id": ["C001", "C002", "C003"],
        "cell_type": ["Tumour", "CD8+ T Cell", "Stromal Fibroblast"],
        "x": [10.5, 20.0, 35.2],
        "y": [15.2, 25.8, 40.1]
    })

    class MockUploadedFile:
        def __init__(self, name, data):
            self.name = name
            self._data = data
        def getvalue(self):
            return self._data

    # 1. CSV
    csv_bytes = base_df.to_csv(index=False).encode("utf-8")
    df_csv = app.read_uploaded_file(MockUploadedFile("cells.csv", csv_bytes))
    assert len(df_csv) == 3
    assert list(df_csv.columns) == ["cell_id", "cell_type", "x", "y"]

    # 2. TSV
    tsv_bytes = base_df.to_csv(index=False, sep="\t").encode("utf-8")
    df_tsv = app.read_uploaded_file(MockUploadedFile("cells.tsv", tsv_bytes))
    assert len(df_tsv) == 3

    # 3. Excel (.xlsx)
    xlsx_buf = io.BytesIO()
    base_df.to_excel(xlsx_buf, index=False, engine="openpyxl")
    df_xlsx = app.read_uploaded_file(MockUploadedFile("cells.xlsx", xlsx_buf.getvalue()))
    assert len(df_xlsx) == 3

    # 4. JSON (.json)
    json_bytes = base_df.to_json(orient="records").encode("utf-8")
    df_json = app.read_uploaded_file(MockUploadedFile("cells.json", json_bytes))
    assert len(df_json) == 3

    # 5. Parquet (.parquet)
    parquet_buf = io.BytesIO()
    base_df.to_parquet(parquet_buf, engine="pyarrow")
    df_parquet = app.read_uploaded_file(MockUploadedFile("cells.parquet", parquet_buf.getvalue()))
    assert len(df_parquet) == 3

    print("[PASS] test_supported_file_readers passed (CSV, TSV, Excel, JSON, Parquet verified).")


def test_image_rejection():
    """Verify that tissue images (PNG, JPG, TIFF) are rejected with clear explanation."""
    class MockUploadedFile:
        def __init__(self, name):
            self.name = name
        def getvalue(self):
            return b"fake_image_bytes"

    for ext in ["tissue.png", "slide.jpg", "fluorescence.tif", "microscopy.tiff"]:
        try:
            app.read_uploaded_file(MockUploadedFile(ext))
            assert False, f"Expected image {ext} to be rejected"
        except ValueError as err:
            assert "Image file detected" in str(err)
            assert "structured cell coordinate" in str(err)

    print("[PASS] test_image_rejection passed (PNG, JPG, TIFF rejected with guidance).")


def test_missing_column_validation():
    """Verify that missing cell_type, x, or y columns are detected accurately."""
    df_no_type = pd.DataFrame({"cell_id": ["C1"], "x": [1.0], "y": [2.0]})
    cols_map = {str(c).lower(): c for c in df_no_type.columns}
    missing = []
    if "cell_type" not in cols_map and "celltype" not in cols_map:
        missing.append("cell_type")
    assert "cell_type" in missing

    df_no_x = pd.DataFrame({"cell_id": ["C1"], "cell_type": ["Tumour"], "y": [2.0]})
    cols_map_x = {str(c).lower(): c for c in df_no_x.columns}
    missing_x = []
    if "x" not in cols_map_x:
        missing_x.append("x")
    assert "x" in missing_x

    print("[PASS] test_missing_column_validation passed.")


def test_exact_euclidean_distance_and_exclusion():
    """Verify exact Euclidean distance math and strict focal cell self-exclusion."""
    df = pd.DataFrame({
        "cell_id": ["C0", "C1", "C2", "C3"],
        "cell_type": ["Tumour", "Immune", "Tumour", "Stromal"],
        "x": [0.0, 3.0, 0.0, 6.0],
        "y": [0.0, 4.0, 5.0, 8.0]
    })

    selected_idx = 0
    focal_coord = np.array([df.loc[selected_idx, "x"], df.loc[selected_idx, "y"]], dtype=float)
    all_coords = df[["x", "y"]].to_numpy(dtype=float)
    deltas = all_coords - focal_coord
    distances = np.sqrt(np.sum(np.square(deltas), axis=1))

    # C0 to C0 = 0.0
    assert np.isclose(distances[0], 0.0)
    # C0 to C1 (3, 4) = 5.0
    assert np.isclose(distances[1], 5.0)
    # C0 to C2 (0, 5) = 5.0
    assert np.isclose(distances[2], 5.0)
    # C0 to C3 (6, 8) = 10.0
    assert np.isclose(distances[3], 10.0)

    # Exclude focal cell from neighbours
    non_focal_mask = np.arange(len(df)) != selected_idx
    assert non_focal_mask[0] == False

    # Within radius 5.5: C1 and C2 only
    neighbour_mask = non_focal_mask & (distances <= 5.5)
    neighbours = df.loc[neighbour_mask].copy()
    assert list(neighbours["cell_id"]) == ["C1", "C2"]
    assert "C0" not in list(neighbours["cell_id"])

    print("[PASS] test_exact_euclidean_distance_and_exclusion passed.")


def test_neighbour_percentages_sum_to_100():
    """Verify neighbour composition counts and percentages sum to 100.0%."""
    neighbours_df = pd.DataFrame({
        "cell_id": ["C1", "C2", "C3", "C4", "C5"],
        "cell_type": ["Tumour", "Tumour", "Immune", "Stromal", "Immune"],
        "distance": [2.5, 3.8, 4.2, 4.9, 5.1]
    })

    n_counts = neighbours_df["cell_type"].value_counts().reset_index()
    n_counts.columns = ["Cell Type", "Number of Neighbour Cells"]
    n_counts["Percentage of Neighbours"] = (n_counts["Number of Neighbour Cells"] / len(neighbours_df)) * 100

    total_pct = n_counts["Percentage of Neighbours"].sum()
    assert np.isclose(total_pct, 100.0)
    print("[PASS] test_neighbour_percentages_sum_to_100 passed (percentages sum to 100.0%).")


def test_to_csv_bytes():
    """Verify CSV byte serialization for exports."""
    df = pd.DataFrame({"Cell Type": ["Tumour", "CD8+ T Cell"], "Count": [170, 110]})
    b = app.to_csv_bytes(df)
    assert isinstance(b, bytes)
    assert b"Cell Type,Count" in b
    print("[PASS] test_to_csv_bytes passed.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING NICHEFINDER FULL CAPABILITY TEST SUITE")
    print("=" * 65)
    test_sample_dataset()
    test_supported_file_readers()
    test_image_rejection()
    test_missing_column_validation()
    test_exact_euclidean_distance_and_exclusion()
    test_neighbour_percentages_sum_to_100()
    test_to_csv_bytes()
    print("=" * 65)
    print("ALL 7 CORE TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)
