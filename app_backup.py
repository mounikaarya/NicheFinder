
import gzip
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="NicheFinder | Spatial Biology",
    page_icon="🧬",
    layout="wide",
)

MAX_CELLS = 10000

COLORS = {
    "Tumour": "#ff6685",
    "Tumor": "#ff6685",
    "Immune": "#62e6d5",
    "Stromal": "#aa91ff",
    "Epithelial": "#ffc66d",
    "Other": "#91a9c4",
    "Unannotated": "#8293a8",
}

# ============================================================
# PREMIUM DARK THEME
# ============================================================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(
        ellipse at top left, #172b46, #0a1220 65%
    );
    color: #edf4ff;
}
section[data-testid="stSidebar"] {
    background: #101c2e;
    border-right: 1px solid #263b53;
}
.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}
h1, h2, h3 {color: #f3f7ff !important;}
p, label {color: #bdcde0;}
.hero {
    font-size: 3.1rem;
    font-weight: 850;
    letter-spacing: -2px;
    line-height: 1.1;
}
.hero span {color: #62e6d5;}
.subtitle {
    color: #a9bed5;
    font-size: 1.1rem;
}
.eyebrow {
    color: #62e6d5;
    letter-spacing: 3px;
    font-size: .8rem;
    font-weight: 800;
}
.metric {
    background: linear-gradient(135deg, #192c44, #111d30);
    border: 1px solid #30445c;
    border-radius: 15px;
    padding: 18px;
    min-height: 105px;
}
.metric-label {
    color: #9bb2cb;
    font-size: .8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    color: white;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 5px;
}
div[data-testid="stMetric"] {
    background: #14243a;
    border: 1px solid #2b4058;
    padding: 15px;
    border-radius: 12px;
}
.stButton button, .stDownloadButton button {
    border-radius: 9px;
    font-weight: 700;
}
hr {border-color: #2b4058;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# COLUMN DETECTION
# ============================================================
ALIASES = {
    "cell_id": [
        "cell_id", "cellid", "cell", "barcode", "cell_barcode",
        "cell_identifier", "cell_name", "id", "object_id",
        "segmentation_id", "feature_id", "obs_id"
    ],
    "cell_type": [
        "cell_type", "celltype", "cell_type_label",
        "celltype_label", "annotation", "annotations",
        "cell_annotation", "celltype_annotation",
        "predicted_cell_type", "predicted_celltype",
        "cluster_annotation", "label", "cell_label",
        "identity", "class", "cell_class"
    ],
    "x": [
        "x", "x_coord", "x_coordinate", "x_centroid",
        "center_x", "centre_x", "centroid_x",
        "spatial_x", "global_x", "image_x",
        "coordinate_x", "x_position", "pos_x"
    ],
    "y": [
        "y", "y_coord", "y_coordinate", "y_centroid",
        "center_y", "centre_y", "centroid_y",
        "spatial_y", "global_y", "image_y",
        "coordinate_y", "y_position", "pos_y"
    ],
}


def normalize_name(name):
    """Normalize column names for comparison."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def detect_column(columns, target):
    """Return a likely column name, or None."""
    aliases = ALIASES[target]
    normalized = {
        col: normalize_name(col)
        for col in columns
    }

    # Exact alias match first.
    for alias in aliases:
        for original, cleaned in normalized.items():
            if cleaned == alias:
                return original

    # Conservative matching for common prefixes/suffixes.
    for original, cleaned in normalized.items():
        if target == "x" and (
            cleaned.endswith("_x") or cleaned.startswith("x_")
        ):
            return original

        if target == "y" and (
            cleaned.endswith("_y") or cleaned.startswith("y_")
        ):
            return original

        if target == "cell_id" and (
            cleaned.endswith("_barcode")
            or cleaned.endswith("_cell_id")
        ):
            return original

    return None


# ============================================================
# FILE READER
# ============================================================
def read_table(uploaded_file):
    """Read supported tabular formats without changing the source."""
    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if filename.endswith(".csv.gz") or filename.endswith(".tsv.gz"):
        content = gzip.decompress(content)
        filename = filename[:-3]

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))

    if filename.endswith(".tsv"):
        return pd.read_csv(
            io.BytesIO(content),
            sep="\t",
        )

    if filename.endswith(".txt"):
        return pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
        )

    if filename.endswith(".csv"):
        return pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
        )

    raise ValueError(
        "Unsupported file. Use CSV, TSV, TXT, XLSX, XLS, "
        "CSV.GZ or TSV.GZ."
    )


# ============================================================
# DEMONSTRATION DATA
# ============================================================
def make_demo_data():
    rng = np.random.default_rng(42)

    groups = [
        ("Tumour", 120, 30, 30, 10, 9),
        ("Immune", 75, 39, 39, 13, 13),
        ("Stromal", 95, 55, 28, 12, 14),
        ("Epithelial", 65, 70, 65, 9, 10),
        ("Other", 45, 18, 70, 10, 8),
    ]

    frames = []
    number = 1

    for label, n, cx, cy, sx, sy in groups:
        x = rng.normal(cx, sx, n)
        y = rng.normal(cy, sy, n)

        frames.append(pd.DataFrame({
            "cell_id": [
                f"DEMO_{number+i:04d}" for i in range(n)
            ],
            "cell_type": label,
            "x": x,
            "y": y,
        }))

        number += n

    return pd.concat(frames, ignore_index=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧬 NicheFinder")
    st.caption("Spatial Biology Explorer")
    st.divider()

    st.markdown("### DATA SOURCE")

    source = st.radio(
        "Choose your data",
        ["Demonstration dataset", "Upload dataset"],
    )

    uploaded = None
    annotation_file = None

    if source == "Upload dataset":
        uploaded = st.file_uploader(
            "Upload spatial data",
            type=["csv", "tsv", "txt", "xlsx", "xls", "gz"],
            help=(
                "Supported: CSV, TSV, TXT, Excel, "
                "CSV.GZ and TSV.GZ."
            ),
        )

        annotation_file = st.file_uploader(
            "Optional: upload a separate cell annotation file",
            type=["csv", "tsv", "txt", "xlsx", "xls", "gz"],
            key="annotation_upload",
            help=(
                "Use this when your coordinate file has cell IDs "
                "but no cell-type labels."
            ),
        )

        with st.expander("Supported column examples"):
            st.write("**Cell ID:**")
            st.code("cell_id, barcode, cell_barcode")

            st.write("**Cell type:**")
            st.code("cell_type, annotation, predicted_cell_type")

            st.write("**Coordinates:**")
            st.code("x, x_centroid, spatial_x")
            st.code("y, y_centroid, spatial_y")

            st.caption(
                "Column names are detected automatically when possible. "
                "You can also select columns manually."
            )

    st.divider()
    st.markdown("### ANALYSIS SETTINGS")

    radius = st.slider(
        "Neighbourhood radius",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
    )

    st.caption(
        "Radius uses the same units as your spatial coordinates."
    )

    st.divider()
    st.caption(
        "Research demonstration only. Not intended for diagnosis."
    )


# ============================================================
# LOAD RAW DATA
# ============================================================
if source == "Demonstration dataset":
    raw_df = make_demo_data()
    is_demo = True

elif uploaded is not None:
    try:
        raw_df = read_table(uploaded)
        is_demo = False

        if raw_df.empty:
            st.error("The uploaded file contains no rows.")
            st.stop()

        if len(raw_df.columns) < 2:
            st.error(
                "The file has too few columns. "
                "Check its format and delimiter."
            )
            st.stop()

    except Exception as error:
        st.error(f"Could not read the uploaded file: {error}")
        st.info(
            "For Excel files, install openpyxl in your environment "
            "if it is missing. For unsupported spatial formats, "
            "export a suitable table from the original software."
        )
        st.stop()

else:
    st.markdown(
        '<div class="eyebrow">SPATIAL BIOLOGY LAB</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero">Discover the '
        '<span>neighbourhoods</span><br>of cells.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Import spatial data, map cellular '
        'locations, and explore local cell-type composition.</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Choose the demonstration dataset or upload your own data "
        "from a compatible spatial biology workflow."
    )
    st.stop()


# ============================================================
# SMART DATASET IMPORT
# ============================================================
st.markdown(
    '<div class="eyebrow">SMART DATASET IMPORT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero">Discover the <span>neighbourhoods</span>'
    '<br>of cells.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Map cellular identities. '
    'Explore neighbouring cell types. Understand tissue organization.'
    '</div>',
    unsafe_allow_html=True,
)

if is_demo:
    st.warning(
        "DEMONSTRATION MODE: These are simulated coordinates. "
        "They are not real patient data or experimental observations."
    )
else:
    st.info(
        "Uploaded dataset detected. Confirm its source, biological "
        "annotations, coordinate system and units before interpretation."
    )

if len(raw_df) > MAX_CELLS:
    st.error(
        f"This demo supports at most {MAX_CELLS:,} rows at a time. "
        "Filter or sample your dataset before uploading."
    )
    st.stop()

if raw_df.columns.duplicated().any():
    st.error(
        "The file contains duplicate column names. Rename them "
        "before continuing so the correct fields can be selected."
    )
    st.stop()

st.markdown("### 1. Preview your original dataset")

st.caption(
    f"File preview: {len(raw_df):,} rows × {len(raw_df.columns):,} columns. "
    "The original upload is not modified."
)

with st.expander("View original columns and data", expanded=False):
    st.write("Available columns:")
    st.code(", ".join(map(str, raw_df.columns)))
    st.dataframe(
        raw_df.head(10),
        use_container_width=True,
        hide_index=True,
    )

# Detect likely columns.
detected = {
    key: detect_column(raw_df.columns, key)
    for key in ["cell_id", "cell_type", "x", "y"]
}

st.markdown("### 2. Map the required columns")

st.write(
    "Check the automatic selections. If your dataset uses different "
    "column names, choose the correct fields below."
)

column_options = ["-- Not available --"] + list(raw_df.columns)


def default_index(value):
    if value in raw_df.columns:
        return list(raw_df.columns).index(value) + 1
    return 0


with st.container(border=True):
    map_a, map_b = st.columns(2)

    with map_a:
        id_col = st.selectbox(
            "Cell ID / barcode",
            column_options,
            index=default_index(detected["cell_id"]),
            key="map_cell_id",
        )

        x_col = st.selectbox(
            "X coordinate",
            column_options,
            index=default_index(detected["x"]),
            key="map_x",
        )

    with map_b:
        type_col = st.selectbox(
            "Cell-type annotation (optional)",
            column_options,
            index=default_index(detected["cell_type"]),
            key="map_cell_type",
        )

        y_col = st.selectbox(
            "Y coordinate",
            column_options,
            index=default_index(detected["y"]),
            key="map_y",
        )

if x_col == "-- Not available --" or y_col == "-- Not available --":
    st.error(
        "Spatial X and Y coordinates are required. "
        "This application cannot reconstruct missing coordinates "
        "from an ordinary gene-expression table."
    )
    st.stop()

if x_col == y_col:
    st.error("X and Y must come from two different columns.")
    st.stop()

if id_col != "-- Not available --" and (
    id_col == x_col or id_col == y_col
):
    st.error("The cell ID column cannot also be a coordinate column.")
    st.stop()

if type_col != "-- Not available --" and (
    type_col == x_col or type_col == y_col or type_col == id_col
):
    st.error(
        "The cell-type column must be different from the ID "
        "and coordinate columns."
    )
    st.stop()


# ============================================================
# STANDARDIZE DATA
# ============================================================
df = pd.DataFrame()

if id_col == "-- Not available --":
    df["cell_id"] = [
        f"ROW_{i+1:06d}" for i in range(len(raw_df))
    ]
    generated_ids = True
else:
    df["cell_id"] = raw_df[id_col].astype("string")
    generated_ids = False

df["x"] = pd.to_numeric(raw_df[x_col], errors="coerce")
df["y"] = pd.to_numeric(raw_df[y_col], errors="coerce")

if type_col == "-- Not available --":
    df["cell_type"] = "Unannotated"
else:
    labels = raw_df[type_col].astype("string").str.strip()
    labels = labels.replace("", pd.NA)
    df["cell_type"] = labels.fillna("Unannotated")

# Optional annotation matching.
annotation_matched = False

if annotation_file is not None:
    try:
        annotation_raw = read_table(annotation_file)

        if annotation_raw.empty:
            st.warning("The annotation file is empty.")
        else:
            st.markdown("### 3. Match a separate annotation file")

            annotation_id_detected = detect_column(
                annotation_raw.columns, "cell_id"
            )
            annotation_type_detected = detect_column(
                annotation_raw.columns, "cell_type"
            )

            annotation_options = list(annotation_raw.columns)

            ann_a, ann_b = st.columns(2)

            with ann_a:
                annotation_id_col = st.selectbox(
                    "Annotation file: cell ID / barcode",
                    annotation_options,
                    index=(
                        annotation_options.index(annotation_id_detected)
                        if annotation_id_detected in annotation_options
                        else 0
                    ),
                    key="annotation_id_column",
                )

            with ann_b:
                annotation_type_col = st.selectbox(
                    "Annotation file: cell type",
                    annotation_options,
                    index=(
                        annotation_options.index(annotation_type_detected)
                        if annotation_type_detected in annotation_options
                        else min(1, len(annotation_options) - 1)
                    ),
                    key="annotation_type_column",
                )

            if annotation_id_col == annotation_type_col:
                st.error(
                    "Annotation IDs and cell-type labels must use "
                    "different columns."
                )
            else:
                annotation_table = pd.DataFrame({
                    "_match_id": (
                        annotation_raw[annotation_id_col]
                        .astype("string").str.strip()
                    ),
                    "_matched_type": (
                        annotation_raw[annotation_type_col]
                        .astype("string").str.strip()
                    ),
                })

                annotation_table = annotation_table.dropna(
                    subset=["_match_id", "_matched_type"]
                )

                annotation_table = annotation_table[
                    (annotation_table["_match_id"] != "")
                    & (annotation_table["_matched_type"] != "")
                ]

                duplicate_keys = annotation_table[
                    "_match_id"
                ].duplicated(keep=False)

                if duplicate_keys.any():
                    st.warning(
                        "The annotation file contains duplicate IDs. "
                        "Only the first valid annotation per ID will "
                        "be used. Review these duplicates before "
                        "interpreting results."
                    )

                annotation_table = annotation_table.drop_duplicates(
                    subset="_match_id",
                    keep="first",
                )

                df["_match_id"] = df["cell_id"].astype("string").str.strip()

                matched = df["_match_id"].isin(
                    annotation_table["_match_id"]
                )

                matched_types = df["_match_id"].map(
                    annotation_table.set_index("_match_id")[
                        "_matched_type"
                    ]
                )

                # Preserve existing labels; fill only missing/unannotated ones.
                needs_label = (
                    df["cell_type"].isna()
                    | (df["cell_type"] == "")
                    | (df["cell_type"] == "Unannotated")
                )

                replace_mask = matched & needs_label
                df.loc[replace_mask, "cell_type"] = matched_types[
                    replace_mask
                ]

                annotation_matched = bool(matched.any())

                st.caption(
                    f"Annotation IDs matched: {int(matched.sum()):,} "
                    f"of {len(df):,} rows. Existing non-empty labels "
                    "were preserved."
                )

                unmatched_count = int((~matched).sum())

                if unmatched_count:
                    st.warning(
                        f"{unmatched_count:,} coordinate rows did not "
                        "match an annotation ID. Check whether the "
                        "barcodes use the same format and sample."
                    )

                df = df.drop(columns=["_match_id"])

    except Exception as error:
        st.error(f"Could not read or match annotation file: {error}")
        st.stop()


# ============================================================
# VALIDATION AND CLEANING
# ============================================================
st.markdown("### 4. Data quality checks")

df["cell_id"] = df["cell_id"].astype("string").str.strip()
df["cell_type"] = (
    df["cell_type"].astype("string").str.strip()
)

df["cell_id"] = df["cell_id"].replace("", pd.NA)
df["cell_type"] = df["cell_type"].replace("", pd.NA)

df["x"] = df["x"].replace([np.inf, -np.inf], np.nan)
df["y"] = df["y"].replace([np.inf, -np.inf], np.nan)

missing_id = int(df["cell_id"].isna().sum())
missing_x = int(df["x"].isna().sum())
missing_y = int(df["y"].isna().sum())

duplicate_ids = int(
    df["cell_id"].duplicated(keep=False).sum()
)

quality_a, quality_b, quality_c, quality_d = st.columns(4)

quality_a.metric("Missing IDs", missing_id)
quality_b.metric("Invalid X", missing_x)
quality_c.metric("Invalid Y", missing_y)
quality_d.metric("Rows with duplicate IDs", duplicate_ids)

# Missing IDs cannot safely be used for annotation matching.
valid_mask = (
    df["cell_id"].notna()
    & df["x"].notna()
    & df["y"].notna()
    & df["cell_type"].notna()
)

removed_rows = int((~valid_mask).sum())

if removed_rows:
    st.warning(
        f"{removed_rows:,} row(s) with missing IDs, missing labels, "
        "or invalid coordinates were removed from the analysis."
    )

df = df.loc[valid_mask].copy()
df = df.reset_index(drop=True)

if df.empty:
    st.error(
        "No valid rows remain. Check the selected columns and "
        "the source file."
    )
    st.stop()

if len(df) < 2:
    st.error(
        "At least two valid rows are required for neighbourhood analysis."
    )
    st.stop()

if duplicate_ids:
    st.warning(
        "Duplicate cell IDs remain in the dataset. Spatial analysis "
        "can continue, but annotation matching may be ambiguous. "
        "Review duplicates before interpreting the results."
    )

if generated_ids:
    st.info(
        "No ID column was selected. Row-based IDs were generated "
        "for this session. They are technical identifiers, not "
        "original biological cell IDs."
    )

# Annotations are considered usable if at least one genuine label exists.
has_annotations = bool(
    (df["cell_type"] != "Unannotated").any()
)

if not has_annotations:
    st.warning(
        "No cell-type annotations are available. Coordinate mapping "
        "and distance calculations remain available, but cell-type "
        "composition analysis is disabled."
    )

df["x"] = df["x"].astype(float)
df["y"] = df["y"].astype(float)

df = df[["cell_id", "cell_type", "x", "y"]].copy()
cell_types = sorted(df["cell_type"].dropna().unique().tolist())

st.success(
    f"Standardized dataset ready: {len(df):,} valid records."
)


# ============================================================
# METRICS
# ============================================================
dominant = (
    df["cell_type"].value_counts().idxmax()
    if has_annotations
    else "Unavailable"
)

a, b, c, d = st.columns(4)

with a:
    st.markdown(
        f'<div class="metric"><div class="metric-label">'
        f'Valid records</div><div class="metric-value">'
        f'{len(df):,}</div></div>',
        unsafe_allow_html=True,
    )

with b:
    st.markdown(
        f'<div class="metric"><div class="metric-label">'
        f'Cell labels</div><div class="metric-value">'
        f'{df.loc[df["cell_type"] != "Unannotated", "cell_type"].nunique()}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

with c:
    st.markdown(
        f'<div class="metric"><div class="metric-label">'
        f'Most frequent label</div><div class="metric-value" '
        f'style="font-size:1.35rem">{dominant}</div></div>',
        unsafe_allow_html=True,
    )

with d:
    st.markdown(
        f'<div class="metric"><div class="metric-label">'
        f'Neighbour radius</div><div class="metric-value">'
        f'{radius:g}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TABS
# ============================================================
tab_map, tab_analysis, tab_data, tab_import, tab_method = st.tabs([
    "◉ Spatial Map",
    "⌘ Neighbourhood Analysis",
    "▤ Data Explorer",
    "⇧ Import & Export",
    "ⓘ Methodology",
])


# ============================================================
# SPATIAL MAP
# ============================================================
with tab_map:
    st.subheader("Spatial cell distribution")

    chosen_types = st.multiselect(
        "Display labels",
        cell_types,
        default=cell_types,
    )

    plot_df = df[df["cell_type"].isin(chosen_types)]

    if plot_df.empty:
        st.info("Select at least one label to display the map.")
    else:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="cell_type",
            hover_name="cell_id",
            color_discrete_map=COLORS,
            render_mode="webgl",
            labels={
                "x": "Spatial X coordinate",
                "y": "Spatial Y coordinate",
                "cell_type": "Cell label",
            },
        )

        fig.update_traces(
            marker=dict(size=8, opacity=0.88)
        )

        fig.update_layout(
            height=550,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,15,26,.7)",
            font_color="#dce8f7",
            legend_title_text="Cell label",
            margin=dict(l=10, r=10, t=25, b=10),
        )

        fig.update_xaxes(gridcolor="#23364c")
        fig.update_yaxes(
            gridcolor="#23364c",
            scaleanchor="x",
            scaleratio=1,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each point represents one row in the standardized dataset. "
        "For spot-based datasets, a point may represent a tissue spot "
        "rather than an individual cell."
    )


# ============================================================
# NEIGHBOURHOOD ANALYSIS
# ============================================================
with tab_analysis:
    st.subheader("Explore a spatial neighbourhood")

    st.write(
        "Choose a focal record to find other records within the "
        "selected Euclidean distance radius."
    )

    chosen_type = st.selectbox(
        "Focal label",
        cell_types,
    )

    possible = df.index[
        df["cell_type"] == chosen_type
    ].tolist()

    focal_idx = st.selectbox(
        "Focal record ID",
        possible,
        format_func=lambda idx: str(df.loc[idx, "cell_id"]),
    )

    coords = df[["x", "y"]].to_numpy(dtype=float)
    focal = coords[focal_idx]

    distances = np.sqrt(
        ((coords - focal) ** 2).sum(axis=1)
    )

    mask = (
        (distances <= radius)
        & (np.arange(len(df)) != focal_idx)
    )

    neighbours = df.loc[
        mask, ["cell_id", "cell_type", "x", "y"]
    ].copy()

    neighbours["distance"] = distances[mask]

    n = len(neighbours)

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Focal record",
        str(df.loc[focal_idx, "cell_id"]),
    )
    m2.metric("Nearby records", n)
    m3.metric(
        "Distinct labels nearby",
        (
            neighbours.loc[
                neighbours["cell_type"] != "Unannotated",
                "cell_type"
            ].nunique()
            if has_annotations else 0
        ),
    )

    if n:
        left, right = st.columns([1.2, 1])

        if has_annotations:
            counts = (
                neighbours["cell_type"]
                .value_counts()
                .rename_axis("cell_type")
                .reset_index(name="count")
            )

            counts["percentage"] = (
                counts["count"] / counts["count"].sum() * 100
            )

            with left:
                bar = px.bar(
                    counts,
                    x="cell_type",
                    y="count",
                    color="cell_type",
                    color_discrete_map=COLORS,
                    title="Nearby record labels",
                    labels={
                        "cell_type": "Cell label",
                        "count": "Number of nearby records",
                    },
                )

                bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,15,26,.7)",
                    font_color="#dce8f7",
                    showlegend=False,
                )

                st.plotly_chart(bar, use_container_width=True)

            with right:
                st.markdown("#### Neighbour composition")

                display = counts.copy()
                display["percentage"] = display[
                    "percentage"
                ].map(lambda value: f"{value:.1f}%")

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.info(
                "Distance-based neighbours are available, but "
                "cell-type composition cannot be calculated without "
                "cell-type annotations."
            )

        with st.expander("View nearby records"):
            st.dataframe(
                neighbours.sort_values("distance"),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download neighbour results",
            neighbours.sort_values("distance").to_csv(
                index=False
            ).encode("utf-8"),
            file_name="nichefinder_neighbours.csv",
            mime="text/csv",
            key="download_neighbours",
        )

    else:
        st.info(
            "No other records were found within this radius. "
            "Try increasing the radius."
        )

    st.caption(
        "Spatial proximity does not prove cell contact, communication, "
        "causation or clinical significance."
    )


# ============================================================
# DATA EXPLORER
# ============================================================
with tab_data:
    st.subheader("Standardized data explorer")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    st.download_button(
        "Download standardized CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="nichefinder_standardized.csv",
        mime="text/csv",
        key="download_standardized",
    )

    if has_annotations:
        composition = (
            df[df["cell_type"] != "Unannotated"]["cell_type"]
            .value_counts()
            .rename_axis("cell_type")
            .reset_index(name="records")
        )

        composition["percentage"] = (
            composition["records"]
            / composition["records"].sum() * 100
        )

        st.subheader("Cell-label composition")

        pie = px.pie(
            composition,
            values="records",
            names="cell_type",
            hole=0.58,
            color="cell_type",
            color_discrete_map=COLORS,
        )

        pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#dce8f7",
        )

        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info(
            "Label composition is unavailable because this dataset "
            "has no validated cell-type annotations."
        )


# ============================================================
# IMPORT / EXPORT AND DATABASE GUIDANCE
# ============================================================
with tab_import:
    st.subheader("Smart Dataset Import")

    st.markdown("""
### What this version supports

- CSV, TSV and delimited TXT files
- Excel XLSX and XLS files, if the required Excel reader is installed
- Gzip-compressed CSV and TSV files
- Automatic column-name detection
- Manual column mapping
- Separate annotation-file matching using IDs or barcodes
- Standardized CSV export

### Preparing data from public databases

**10x Genomics Xenium:** look for cell-level coordinate and annotation tables. Export or select the appropriate cell table and map its centroid X/Y columns.

**10x Genomics Visium:** many datasets contain tissue spots rather than segmented individual cells. NicheFinder can map spot coordinates, but those records must not be described as individual cells.

**Other spatial platforms:** export a table containing two spatial coordinates per observation. Use a validated annotation table if cell-type labels are stored separately.

### Standard output format

""")

    st.code(
        "cell_id,cell_type,x,y\n"
        "CELL_001,Immune,12.4,18.2\n"
        "CELL_002,Tumour,13.1,19.0",
        language="csv",
    )

    st.download_button(
        "Download standardized dataset",
        df.to_csv(index=False).encode("utf-8"),
        file_name="nichefinder_standardized.csv",
        mime="text/csv",
        key="download_import_tab",
    )

    st.markdown("### Important limitations")

    st.warning(
        "This app does not connect directly to every public database, "
        "parse every specialized spatial format, identify cells from "
        "microscopy images, or infer missing biological annotations. "
        "Some datasets require preprocessing or export from their "
        "original software."
    )


# ============================================================
# METHODOLOGY
# ============================================================
with tab_method:
    st.subheader("How NicheFinder works")

    st.markdown("""
**1. Data import**

The app reads supported tabular formats, detects likely column names,
and lets the user map cell IDs, labels and spatial coordinates.

**2. Data standardization**

Compatible data are converted to `cell_id`, `cell_type`, `x`, and `y`.
Rows with missing IDs or invalid coordinates are excluded from analysis.

**3. Spatial distance**

For records *i* and *j*, Euclidean distance is:

`distance = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)`

**4. Neighbourhood definition**

Other records within the selected radius are counted as neighbours.

**5. Composition analysis**

When valid labels are available, the app summarizes the labels among
nearby records and calculates their percentages.

**6. Annotation matching**

A separate annotation file can be joined using matching cell IDs or
barcodes. Unmatched records are reported instead of being assigned
invented biological labels.

**Limitations**

Results depend on dataset quality, annotation reliability, coordinate
units and the selected radius. The app does not identify cells from
images, infer cell communication, or establish clinical significance.
""")

    if is_demo:
        st.warning(
            "The current dataset is simulated. Its patterns must not "
            "be interpreted as real biological findings."
        )
    else:
        st.info(
            "The app does not independently verify data provenance, "
            "cell identities, experimental quality or coordinate units."
        )


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.markdown(
    "<div style='text-align:center;color:#91a5be;'>"
    "🧬 NicheFinder · Spatial Biology Explorer<br>"
    "Smart Dataset Import · Research demonstration only · Not for diagnosis"
    "</div>",
    unsafe_allow_html=True,
)
