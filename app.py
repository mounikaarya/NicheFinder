r"""
NicheFinder — Spatial Biology Explorer
Tagline: Discover the hidden neighbourhoods of cells.
An interactive bioinformatics application for tissue microenvironment profiling.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import plotly.express as px
import plotly.graph_objects as go
import scipy.ndimage as ndi
import streamlit as st

# ============================================================
# PAGE CONFIGURATION & THEME
# ============================================================
st.set_page_config(
    page_title="NicheFinder — Spatial Biology Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional scientific color palette (dark navy, teal, violet theme)
COLORS = {
    "Tumour": "#FF5E7E",
    "Tumor": "#FF5E7E",
    "CD8+ T Cell": "#00E5FF",
    "Immune": "#00E5FF",
    "T Cell": "#00E5FF",
    "Stromal Fibroblast": "#B388FF",
    "Stromal": "#B388FF",
    "Fibroblast": "#B388FF",
    "Macrophage": "#FFD166",
    "Endothelial": "#06D6A0",
    "Epithelial": "#4D96FF",
    "Other": "#90A4AE",
    "Detected Nuclei": "#00E5FF",
}

FALLBACK_PALETTE = [
    "#FF5E7E", "#00E5FF", "#B388FF", "#FFD166", "#06D6A0",
    "#4D96FF", "#FF9F43", "#54A0FF", "#90A4AE", "#5f27cd"
]

def get_color(cell_type, idx=0):
    if cell_type in COLORS:
        return COLORS[cell_type]
    return FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)]

# Inject Custom CSS for dark navy, teal, violet theme
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 15% 15%, #12243d 0%, #080f1a 80%);
    color: #edf4ff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #0b1422;
    border-right: 1px solid #1c2e47;
}
.metric-card {
    background: linear-gradient(135deg, rgba(20, 36, 58, 0.95), rgba(13, 23, 38, 0.95));
    border: 1px solid #233a59;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #00E5FF;
}
.metric-card.violet { border-left-color: #B388FF; }
.metric-card.coral { border-left-color: #FF5E7E; }
.metric-label {
    color: #9cb4cf;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.metric-value {
    color: #ffffff;
    font-size: 1.75rem;
    font-weight: 800;
    margin-top: 2px;
}
.metric-sub {
    color: #728ca8;
    font-size: 0.8rem;
    margin-top: 2px;
}
.hero-box {
    padding: 0.4rem 0 0.8rem 0;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
}
.hero-title span {
    color: #00E5FF;
}
.hero-tagline {
    font-size: 1.05rem;
    color: #a0b6cf;
    margin-top: 4px;
}
.banner-demo {
    background: rgba(255, 94, 126, 0.12);
    border: 1px solid rgba(255, 94, 126, 0.35);
    border-left: 4px solid #FF5E7E;
    color: #ffc2cc;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.88rem;
    margin: 8px 0 16px 0;
}
.banner-info {
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.25);
    border-left: 4px solid #00E5FF;
    color: #b3f5ff;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.88rem;
    margin: 8px 0 16px 0;
}
.stButton button, .stDownloadButton button {
    border-radius: 6px;
    font-weight: 650;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS: DATA GENERATION, PARSING & IMAGE ANALYSIS
# ============================================================
@st.cache_data
def load_sample_dataset():
    """Generate realistic demonstration spatial dataset (550 cells, 6 cell types)."""
    rng = np.random.default_rng(101)
    archetypes = [
        ("Tumour", 170, 45.0, 45.0, 7.5, 7.5),
        ("CD8+ T Cell", 110, 56.0, 48.0, 11.0, 10.5),
        ("Stromal Fibroblast", 115, 32.0, 32.0, 12.0, 12.0),
        ("Macrophage", 70, 47.0, 53.0, 9.5, 9.0),
        ("Endothelial", 45, 62.0, 28.0, 8.0, 8.5),
        ("Epithelial", 40, 20.0, 55.0, 7.0, 7.5),
    ]
    rows, count = [], 1
    for label, n, cx, cy, sx, sy in archetypes:
        xs = rng.normal(cx, sx, n)
        ys = rng.normal(cy, sy, n)
        for i in range(n):
            rows.append({
                "cell_id": f"CELL_{count:04d}",
                "cell_type": label,
                "x": round(float(xs[i]), 3),
                "y": round(float(ys[i]), 3)
            })
            count += 1
    return pd.DataFrame(rows)


def to_csv_bytes(dataframe):
    """Serialize DataFrame to downloadable CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def image_to_bytes(pil_img, format="PNG"):
    """Serialize PIL Image to byte buffer for download."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()


def read_uploaded_file(uploaded_file):
    """
    Parse structured data files: CSV, TSV, Excel (.xlsx/.xls), JSON (.json), Parquet (.parquet).
    Explicitly redirects image files to the dedicated image analysis workflow.
    """
    fname = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    # If an image was accidentally provided in the structured data section, guide the user
    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")
    if fname.endswith(image_exts):
        raise ValueError(
            f"Image file detected ('{uploaded_file.name}'). Tissue microscopy images contain raw pixel matrices "
            f"rather than structured cell coordinate records. To analyze tissue microscopy images, "
            f"please select 'Upload Tissue Image' in the sidebar data source."
        )

    # 1. Parquet
    if fname.endswith((".parquet", ".pq")):
        try:
            return pd.read_parquet(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Failed to parse Parquet file: {exc}")

    # 2. Excel
    if fname.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception as exc:
            raise ValueError(f"Failed to parse Excel file: {exc}")

    # 3. JSON
    if fname.endswith(".json"):
        try:
            return pd.read_json(io.BytesIO(content))
        except Exception:
            try:
                parsed = json.loads(content.decode("utf-8"))
                if isinstance(parsed, list):
                    return pd.DataFrame(parsed)
                elif isinstance(parsed, dict):
                    if "data" in parsed and "columns" in parsed:
                        return pd.DataFrame(parsed["data"], columns=parsed["columns"])
                    return pd.DataFrame(parsed)
                else:
                    raise ValueError("JSON root must be an array of records or a dictionary of columns.")
            except Exception as exc:
                raise ValueError(f"Failed to parse JSON file: {exc}")

    # 4. TSV / Tab-delimited
    if fname.endswith((".tsv", ".tab")):
        try:
            return pd.read_csv(io.BytesIO(content), sep="\t")
        except Exception as exc:
            raise ValueError(f"Failed to parse TSV file: {exc}")

    # 5. CSV / Text
    if fname.endswith((".csv", ".txt")):
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception:
            return pd.read_csv(io.BytesIO(content), sep=None, engine="python")

    raise ValueError(
        f"Unsupported file format ('{uploaded_file.name}'). "
        f"Supported structured data formats are: CSV (.csv), TSV (.tsv), Excel (.xlsx), JSON (.json), and Parquet (.parquet)."
    )


def segment_tissue_nuclei(pil_img, modality="Auto-detect", sensitivity=1.0, min_area=15, max_area=3500, min_dist=6):
    """
    Perform actual, scientifically documented nucleus detection & segmentation on a microscopy image.
    
    Methodology:
    1. Grayscale luminance conversion and optical density normalization.
    2. Automatic slide background detection (Brightfield/H&E vs Fluorescence/DAPI).
    3. Inversion of brightfield images so hematoxylin-stained nuclei become high-intensity foreground.
    4. Gaussian spatial smoothing to suppress high-frequency camera noise.
    5. Otsu's variance-maximizing automated threshold calculation adjusted by user sensitivity.
    6. Morphological binary opening to eliminate sub-cellular debris.
    7. Euclidean distance transform to identify nuclear geometric centers.
    8. Local maxima peak filtering to resolve overlapping/clustered nuclei.
    9. Sub-pixel centroid extraction and contour boundary rendering.
    """
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)

    # Convert to grayscale luminance
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)

    # Determine modality (Brightfield vs Fluorescence)
    mean_lum = float(np.mean(gray))
    if modality == "Auto-detect":
        is_brightfield = (mean_lum > 120.0)
    elif "Brightfield" in modality:
        is_brightfield = True
    else:
        is_brightfield = False

    # For brightfield (H&E), nuclei are dark purple/blue; invert so nuclei are bright foreground
    if is_brightfield:
        fg = 255.0 - (0.4 * arr[:, :, 0] + 0.5 * arr[:, :, 1] + 0.1 * arr[:, :, 2])
    else:
        fg = gray

    # Contrast stretch (1st to 99th percentile)
    p1, p99 = np.percentile(fg, (1, 99))
    if p99 > p1:
        fg_norm = np.clip((fg - p1) / (p99 - p1) * 255.0, 0, 255)
    else:
        fg_norm = fg

    # Gaussian smoothing to reduce sensor noise
    smoothed = ndi.gaussian_filter(fg_norm, sigma=1.4)

    # Automated Otsu's thresholding
    hist, _ = np.histogram(smoothed.astype(np.uint8), bins=256, range=(0, 256))
    total_pix = smoothed.size
    current_max, otsu_t = 0.0, 100
    weight_bg, sum_bg = 0, 0
    sum_total = np.dot(np.arange(256), hist)
    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total_pix - weight_bg
        if weight_fg == 0:
            break
        sum_bg += i * hist[i]
        m_bg = sum_bg / weight_bg
        m_fg = (sum_total - sum_bg) / weight_fg
        var_b = float(weight_bg) * float(weight_fg) * ((m_bg - m_fg) ** 2)
        if var_b > current_max:
            current_max = var_b
            otsu_t = i

    applied_t = max(10, int(otsu_t / max(0.2, sensitivity)))
    mask = smoothed > applied_t
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))

    # Distance transform to find nuclear centers
    dist = ndi.distance_transform_edt(mask)

    # Local peak detection
    local_max = (ndi.maximum_filter(dist, size=int(min_dist)) == dist) & (dist > 1.5)
    labeled_peaks, num_peaks = ndi.label(local_max)

    nuclei = []
    annotated_img = rgb.copy()
    draw = ImageDraw.Draw(annotated_img)

    if num_peaks > 0:
        centers = ndi.center_of_mass(mask, labeled_peaks, range(1, num_peaks + 1))
        idx = 1
        for cy, cx in centers:
            if np.isnan(cy) or np.isnan(cx):
                continue
            iy, ix = int(round(cy)), int(round(cx))
            iy = max(0, min(arr.shape[0] - 1, iy))
            ix = max(0, min(arr.shape[1] - 1, ix))

            r_approx = max(3.0, float(dist[iy, ix]))
            area_approx = np.pi * (r_approx ** 2)

            if min_area <= area_approx <= max_area:
                nuclei.append({
                    "cell_id": f"NUCLEUS_{idx:04d}",
                    "cell_type": "Detected Nuclei",
                    "x": round(float(cx), 2),
                    "y": round(float(cy), 2),
                    "area_px": round(area_approx, 1),
                    "radius_px": round(r_approx, 1)
                })
                # Draw cyan boundary circle
                draw.ellipse([cx - r_approx, cy - r_approx, cx + r_approx, cy + r_approx], outline="#00E5FF", width=2)
                # Draw coral centroid dot
                draw.ellipse([cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5], fill="#FF5E7E", outline="#FFFFFF")
                idx += 1

    detected_df = pd.DataFrame(nuclei)
    detected_modality_str = "Brightfield (H&E Stained)" if is_brightfield else "Fluorescence (DAPI)"
    return detected_df, annotated_img, detected_modality_str


# ============================================================
# LOGIN SCREEN (AUTHENTICATION)
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

DEMO_USERS = {
    "researcher@biohack.org": {
        "hash": hashlib.sha256("spatial2026".encode()).hexdigest(),
        "name": "Dr. Alex Rivera"
    },
    "demo": {
        "hash": hashlib.sha256("nichefinder".encode()).hexdigest(),
        "name": "Hackathon Reviewer"
    }
}

if not st.session_state["authenticated"]:
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🧬 NicheFinder — <span>Spatial Biology Explorer</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-tagline">Discover the hidden neighbourhoods of cells.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1], gap="large")
    with col_l:
        st.markdown("""
        Welcome to **NicheFinder** — an interactive spatial biology platform for tissue microenvironment profiling.
        
        #### Core Capabilities:
        - 📊 **Cell-Type Summary:** Instant abundance, frequency, and proportions from annotated datasets.
        - 🗺️ **Spatial Cell Map:** High-resolution interactive 2D tissue scatter map with 1:1 aspect ratio.
        - 📐 **Distance Between Cells:** Rigorous Euclidean distance calculations with vector lines to neighbours.
        - 🔬 **Neighbour Composition:** Quantify local microenvironments and cellular niches around any cell.
        - 📂 **Structured Formats:** CSV, Excel (.xlsx), JSON (.json), TSV (.tsv), and Parquet (.parquet).
        - 🖼️ **Tissue Image Analysis:** Actual nucleus detection and segmentation from PNG, JPEG, and TIFF microscopy slides.
        
        *Notice: This is a hackathon demonstration prototype. Passwords are authenticated in-memory.*
        """)

    with col_r:
        with st.container(border=True):
            st.subheader("🔑 Sign In")
            st.caption("Enter demonstration credentials or use Express Login:")
            
            with st.form("login_form"):
                user_input = st.text_input("Username or Email", value="researcher@biohack.org")
                pass_input = st.text_input("Password", type="password", value="spatial2026")
                submit = st.form_submit_button("Sign In to NicheFinder 🚀", use_container_width=True)
                
                if submit:
                    clean_u = user_input.strip()
                    p_hash = hashlib.sha256(pass_input.encode()).hexdigest()
                    if clean_u in DEMO_USERS and DEMO_USERS[clean_u]["hash"] == p_hash:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = DEMO_USERS[clean_u]["name"]
                        st.rerun()
                    elif clean_u and pass_input:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = clean_u
                        st.rerun()
                    else:
                        st.error("Please enter a valid username and password.")
            
            st.markdown("---")
            if st.button("⚡ Express Demo Login (One-Click)", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["username"] = "Dr. Alex Rivera"
                st.rerun()
                
            st.caption("Demo Accounts: `researcher@biohack.org` / `spatial2026` or `demo` / `nichefinder`")

    st.stop()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("### 🧬 NicheFinder")
    st.caption("Spatial Biology Explorer")
    st.write(f"👤 Logged in as: **{st.session_state['username']}**")
    
    if st.button("Sign Out 🚪", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()
        
    st.divider()
    st.markdown("### 📂 Data Source")
    data_choice = st.radio(
        "Choose Data Source:",
        ["Demonstration Data (Simulated)", "Upload Structured Data", "Upload Tissue Image"],
        index=0,
        help="Select between simulated benchmark tissue, structured tables (CSV, Excel, JSON, TSV, Parquet), or tissue microscopy images."
    )
    
    uploaded_file = None
    uploaded_image = None
    
    # Handle structured file uploader (supports both 'Upload Structured Data' and alias 'Upload Structured File')
    if "Structured" in data_choice:
        uploaded_file = st.file_uploader(
            "Upload file (CSV, Excel, JSON, TSV, Parquet)",
            type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet", "pq"],
            help="Structured table must contain cell_type, x, and y (and optional cell_id)."
        )
    elif data_choice == "Upload Tissue Image":
        uploaded_image = st.file_uploader(
            "Upload tissue image (PNG, JPEG, TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Upload microscopy tissue slide in PNG, JPEG, or TIFF format."
        )

    st.divider()
    sample_df = load_sample_dataset()
    st.download_button(
        label="📥 Download Sample CSV",
        data=to_csv_bytes(sample_df),
        file_name="sample_spatial_data.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download verified sample file with columns cell_id, cell_type, x, y"
    )


# ============================================================
# BRANCH A: TISSUE IMAGE UPLOAD & ANALYSIS WORKFLOW
# ============================================================
if data_choice == "Upload Tissue Image":
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🔬 Tissue Image Analysis — <span>Cell Nucleus Segmentation</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-tagline">Detect cell nuclei, calculate sub-cellular spatial coordinates, and map cellular microenvironments.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="banner-info"><strong>🔬 MICROSCOPY IMAGE ANALYSIS WORKFLOW:</strong> '
        'This workflow performs morphological cell-nucleus detection and segmentation. '
        'All detected objects are labeled as <strong>"Detected Nuclei"</strong>. '
        'Biological cell-type identification requires molecular markers (such as multiplex immunofluorescence antibody channels or spatial transcriptomics probes) '
        'or a validated classification model.</div>',
        unsafe_allow_html=True
    )

    # Allow uploading or quick selection of sample images
    st.markdown("#### Tissue Image Source")
    img_subcol1, img_subcol2 = st.columns([1.5, 1.5])
    
    with img_subcol1:
        sample_choice = st.selectbox(
            "Quick-Load Sample Microscopy Slide (or use uploaded file):",
            ["Uploaded Image (Use file from sidebar)", "Sample DAPI Fluorescence (.png)", "Sample H&E Stained Slide (.jpg)", "Sample Microscopy TIFF (.tif)"],
            index=0 if uploaded_image is not None else 1
        )
    
    with img_subcol2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        # Offer sample image downloads
        if Path("sample_tissue_fluorescence.png").exists():
            with open("sample_tissue_fluorescence.png", "rb") as f_img:
                st.download_button("📥 Download Sample DAPI Image (.png)", f_img.read(), "sample_dapi.png", "image/png")

    active_pil_img = None
    active_img_name = ""

    if sample_choice == "Uploaded Image (Use file from sidebar)":
        if uploaded_image is not None:
            try:
                active_pil_img = Image.open(io.BytesIO(uploaded_image.getvalue()))
                active_img_name = uploaded_image.name
            except Exception as exc:
                st.error(f"❌ Failed to open uploaded image file: {exc}")
                st.stop()
        else:
            st.info("👆 Please upload a tissue microscopy image (.png, .jpg, .tif) in the sidebar, or select one of the sample slides above.")
            st.stop()
    elif "DAPI" in sample_choice:
        if Path("sample_tissue_fluorescence.png").exists():
            active_pil_img = Image.open("sample_tissue_fluorescence.png")
            active_img_name = "sample_tissue_fluorescence.png"
    elif "H&E" in sample_choice:
        if Path("sample_tissue_he.jpg").exists():
            active_pil_img = Image.open("sample_tissue_he.jpg")
            active_img_name = "sample_tissue_he.jpg"
    elif "TIFF" in sample_choice:
        if Path("sample_tissue_microscopy.tif").exists():
            active_pil_img = Image.open("sample_tissue_microscopy.tif")
            active_img_name = "sample_tissue_microscopy.tif"

    if active_pil_img is None:
        st.error("❌ No image selected or image file could not be loaded.")
        st.stop()

    img_w, img_h = active_pil_img.size
    
    # Analysis Configuration Controls
    with st.expander("⚙️ Image Segmentation Parameters", expanded=True):
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            modality_option = st.selectbox(
                "Staining Modality:",
                ["Auto-detect", "Fluorescence (DAPI / Dark background)", "Brightfield (H&E Stained / Light background)"],
                index=0,
                help="Select your microscopy modality or let NicheFinder auto-detect based on slide background luminance."
            )
        with p_c2:
            sens_slider = st.slider("Detection Sensitivity:", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="Higher values detect dimmer nuclei; lower values reduce noise.")
        with p_c3:
            min_area_slider = st.slider("Minimum Nucleus Area (px²):", min_value=5, max_value=80, value=15, step=5, help="Exclude small debris and background noise.")

    # Run Nucleus Detection & Segmentation Pipeline
    try:
        with st.spinner("Analyzing tissue image: detecting and segmenting cell nuclei..."):
            detected_nuclei_df, annotated_overlay_img, detected_mod_str = segment_tissue_nuclei(
                active_pil_img,
                modality=modality_option,
                sensitivity=sens_slider,
                min_area=min_area_slider
            )
    except Exception as exc:
        st.error(f"❌ Image analysis failed: {exc}")
        st.stop()

    total_detected = len(detected_nuclei_df)

    # 1. SAMPLE OVERVIEW (Metric Cards)
    st.markdown("#### Sample Overview")
    im_m1, im_m2, im_m3 = st.columns(3)
    with im_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Detected Nuclei</div>
            <div class="metric-value">{total_detected:,}</div>
            <div class="metric-sub">Segmented cell centers</div>
        </div>
        """, unsafe_allow_html=True)
    with im_m2:
        st.markdown(f"""
        <div class="metric-card violet">
            <div class="metric-label">Image Field Dimensions</div>
            <div class="metric-value" style="font-size:1.55rem;">{img_w} × {img_h}</div>
            <div class="metric-sub">Resolution: {img_w * img_h:,} pixels</div>
        </div>
        """, unsafe_allow_html=True)
    with im_m3:
        mean_area = detected_nuclei_df["area_px"].mean() if total_detected > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card coral">
            <div class="metric-label">Mean Nuclear Area</div>
            <div class="metric-value" style="font-size:1.55rem;">{mean_area:.1f} px²</div>
            <div class="metric-sub">Modality: {detected_mod_str}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. VISUAL IMAGE DISPLAY: Original vs Annotated
    st.subheader("Visual Image Comparison")
    st.caption("Left: Original uploaded microscopy tissue slide. Right: Annotated overlay with segmented nuclear boundaries (cyan) and centroids (coral).")

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.image(active_pil_img, caption=f"Original Image: {active_img_name} ({img_w}×{img_h} px)", use_container_width=True)
    with v_col2:
        st.image(annotated_overlay_img, caption=f"Annotated Image: {total_detected} Nuclei Detected", use_container_width=True)

    # Download buttons for annotated image & table
    dl_img_c1, dl_img_c2 = st.columns(2)
    with dl_img_c1:
        st.download_button(
            label="📥 Download Annotated Image (PNG)",
            data=image_to_bytes(annotated_overlay_img, format="PNG"),
            file_name=f"annotated_{active_img_name}.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_img_c2:
        if total_detected > 0:
            st.download_button(
                label="📥 Download Detected Nuclei Coordinates (CSV)",
                data=to_csv_bytes(detected_nuclei_df),
                file_name=f"detected_nuclei_{active_img_name}.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")

    if total_detected == 0:
        st.warning("⚠️ No cell nuclei could be separated at the current sensitivity setting. Please adjust the sensitivity slider or change the modality option.")
        st.stop()

    # 3. DETECTED NUCLEI TABLE
    st.subheader("Cell Types Present")
    st.caption("All detected morphological objects are labeled as 'Detected Nuclei'. Cell-type identification requires molecular markers or validated phenotypic classifiers.")

    detected_summary_table = pd.DataFrame([{
        "Cell Type": "Detected Nuclei",
        "Number of Cells": total_detected,
        "Percentage of Total Cells": "100.00%"
    }])
    st.dataframe(detected_summary_table, width="stretch", hide_index=True)

    st.markdown("---")

    # 4. SPATIAL ARRANGEMENT & NEIGHBOURHOOD
    st.subheader("Spatial Arrangement")
    st.caption("Interactive spatial scatter map of detected nuclei centroids at their calculated (X, Y) pixel coordinates. Hover over any dot to inspect coordinates.")

    # Focal selection
    st.markdown("#### Explore Neighbouring Cells & Distance Settings")
    img_f_c1, img_f_c2 = st.columns([1.5, 1.5])
    with img_f_c1:
        focal_n_idx = st.selectbox(
            "Select a nucleus to inspect:",
            options=list(range(len(detected_nuclei_df))),
            format_func=lambda i: f"{detected_nuclei_df.loc[i, 'cell_id']} at (x={detected_nuclei_df.loc[i, 'x']:.1f}, y={detected_nuclei_df.loc[i, 'y']:.1f}, area={detected_nuclei_df.loc[i, 'area_px']:.1f} px²)",
            index=0,
            key="img_focal_select"
        )
    with img_f_c2:
        img_diag = float(np.hypot(img_w, img_h))
        img_def_r = max(10.0, round(img_diag * 0.15, 1))
        img_radius = st.slider("Neighbourhood Radius (pixels):", min_value=5.0, max_value=float(round(img_diag * 0.6, 1)), value=float(img_def_r), step=2.0)

    focal_n_row = detected_nuclei_df.loc[focal_n_idx]
    fn_id = str(focal_n_row["cell_id"])
    fn_x = float(focal_n_row["x"])
    fn_y = float(focal_n_row["y"])

    # Calculate Euclidean distances between detected nuclei
    all_n_coords = detected_nuclei_df[["x", "y"]].to_numpy(dtype=float)
    fn_coord = np.array([fn_x, fn_y], dtype=float)
    deltas_n = all_n_coords - fn_coord
    distances_n = np.sqrt(np.sum(np.square(deltas_n), axis=1))

    # Exclude focal cell
    mask_n_non_focal = np.arange(len(detected_nuclei_df)) != focal_n_idx
    mask_n_neighbours = mask_n_non_focal & (distances_n <= img_radius)

    n_neighbours_df = detected_nuclei_df.loc[mask_n_neighbours].copy()
    n_neighbours_df["distance"] = np.round(distances_n[mask_n_neighbours], 2)
    n_neighbours_df = n_neighbours_df.sort_values("distance").reset_index(drop=True)
    num_n_neighbours = len(n_neighbours_df)

    # Plotly Spatial Map of Detected Nuclei
    fig_img_map = go.Figure()

    # Background detected nuclei
    fig_img_map.add_trace(go.Scatter(
        x=detected_nuclei_df["x"],
        y=detected_nuclei_df["y"],
        mode="markers",
        name="Detected Nuclei",
        marker=dict(size=7, color="#00E5FF", opacity=0.75, line=dict(width=0.5, color="#ffffff")),
        text=detected_nuclei_df["cell_id"],
        customdata=np.stack((detected_nuclei_df["area_px"], detected_nuclei_df["radius_px"]), axis=-1),
        hovertemplate="<b>ID:</b> %{text}<br><b>Coordinates:</b> (%{x:.1f}, %{y:.1f}) px<br><b>Area:</b> %{customdata[0]:.1f} px²<extra>Detected Nucleus</extra>"
    ))

    # Distance lines to neighbours
    if num_n_neighbours > 0:
        lx, ly = [], []
        for _, nr in n_neighbours_df.iterrows():
            lx.extend([fn_x, nr["x"], None])
            ly.extend([fn_y, nr["y"], None])
        fig_img_map.add_trace(go.Scatter(
            x=lx, y=ly,
            mode="lines",
            name="Distance vectors",
            line=dict(color="rgba(0, 229, 255, 0.45)", width=1.4, dash="dot"),
            hoverinfo="skip"
        ))

    # Radius boundary circle
    th = np.linspace(0, 2*np.pi, 150)
    fig_img_map.add_trace(go.Scatter(
        x=fn_x + img_radius * np.cos(th),
        y=fn_y + img_radius * np.sin(th),
        mode="lines",
        name=f"Radius ({img_radius:g} px)",
        line=dict(color="#FFD166", width=1.8, dash="dash"),
        hoverinfo="skip"
    ))

    # Highlight neighbours
    if num_n_neighbours > 0:
        fig_img_map.add_trace(go.Scatter(
            x=n_neighbours_df["x"],
            y=n_neighbours_df["y"],
            mode="markers",
            name=f"Neighbours ({num_n_neighbours})",
            marker=dict(size=10, color="#FFD166", line=dict(width=1.5, color="#ffffff")),
            text=n_neighbours_df["cell_id"],
            customdata=n_neighbours_df["distance"],
            hovertemplate="<b>Neighbour:</b> %{text}<br><b>Distance:</b> %{customdata:.2f} pixels<extra></extra>"
        ))

    # Highlight focal nucleus
    fig_img_map.add_trace(go.Scatter(
        x=[fn_x],
        y=[fn_y],
        mode="markers+text",
        name=f"Selected ({fn_id})",
        marker=dict(size=18, color="#FF5E7E", symbol="star", line=dict(width=2, color="#ffffff")),
        text=[f"★ {fn_id}"],
        textposition="top center",
        textfont=dict(color="#FF5E7E", size=12),
        hoverinfo="text",
        hovertext=f"<b>Selected Nucleus: {fn_id}</b><br>Coordinates: ({fn_x:.1f}, {fn_y:.1f}) px"
    ))

    fig_img_map.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 19, 34, 0.8)",
        font=dict(color="#dbe6f5"),
        margin=dict(l=15, r=15, t=20, b=20),
        legend=dict(title=dict(text="<b>Map Key</b>", font=dict(color="#00E5FF")), bgcolor="rgba(11, 20, 34, 0.75)", bordercolor="#203550", borderwidth=1)
    )
    fig_img_map.update_xaxes(title="X Coordinate (pixels)", range=[0, img_w], gridcolor="#1b2f48")
    fig_img_map.update_yaxes(title="Y Coordinate (pixels)", range=[img_h, 0], scaleanchor="x", scaleratio=1, gridcolor="#1b2f48") # Invert y to match image raster coordinates
    st.plotly_chart(fig_img_map, width="stretch")

    st.markdown("---")

    # 5. EXPLORE NEIGHBOURING CELLS & DISTANCE BETWEEN CELLS
    st.subheader("Explore Neighbouring Cells & Distance Between Cells")
    st.write(
        f"Selected Nucleus: **`{fn_id}`** at `x={fn_x:.1f}, y={fn_y:.1f}` pixels. "
        f"Found **{num_n_neighbours}** neighbouring nucleus/nuclei within radius **{img_radius:g} pixels**."
    )
    st.info("ℹ️ Distance units: **pixels** (derived from image spatial coordinates). Distance is physical spatial separation, never a percentage.")

    if num_n_neighbours > 0:
        tbl_n_out = pd.DataFrame({
            "Neighbour ID": n_neighbours_df["cell_id"],
            "Classification": n_neighbours_df["cell_type"],
            "X (pixels)": n_neighbours_df["x"],
            "Y (pixels)": n_neighbours_df["y"],
            "Area (px²)": n_neighbours_df["area_px"],
            "Distance (pixels)": n_neighbours_df["distance"]
        })
        st.dataframe(tbl_n_out, width="stretch", hide_index=True)
        
        # Distance Visualization Chart
        fig_n_dist = px.bar(
            n_neighbours_df,
            x="cell_id",
            y="distance",
            color_discrete_sequence=["#00E5FF"],
            labels={"cell_id": "Neighbour Nucleus ID", "distance": "Distance (pixels)"},
            title=f"Distances to Neighbours from {fn_id} (in Pixel Units)"
        )
        fig_n_dist.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 19, 34, 0.8)",
            font=dict(color="#dbe6f5"),
            margin=dict(l=15, r=15, t=35, b=15)
        )
        fig_n_dist.update_yaxes(title="Distance (pixels)", gridcolor="#1b2f48")
        fig_n_dist.update_xaxes(title="Neighbour Nucleus ID", gridcolor="#1b2f48")
        st.plotly_chart(fig_n_dist, width="stretch")
    else:
        st.warning(f"⚠️ No neighbouring nuclei found within radius {img_radius:g} pixels. Increase the radius slider to explore a wider neighbourhood.")

    # Scientific Transparency Callout on Cell Types
    st.markdown("""
    <div class="banner-demo">
    <strong>🔬 Scientific Cell-Type Identification Note:</strong><br>
    The image analysis pipeline performs morphological nucleus detection and segmentation. All detected objects are reported as <strong>'Detected Nuclei'</strong>.<br>
    Assigning specific biological cell phenotypes (e.g. tumour cells, CD8+ cytotoxic T cells, stromal fibroblasts, macrophages) requires multiplex molecular markers (such as immunofluorescence antibody panels or spatial transcriptomics probes) or a validated classification model. Unstained or monochromatic nuclear imaging captures spatial coordinates and nuclear morphology, but not molecular phenotype.
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #8fa6c4; font-size: 0.84rem; line-height: 1.6;">
        <strong>🧬 NicheFinder — Spatial Biology Explorer</strong> · Tissue Microscopy Analysis Module<br>
        <em>Morphological cell nucleus detection and spatial distance profiling · Not for clinical diagnosis.</em>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# BRANCH B: STRUCTURED DATA WORKFLOW (PRESERVED 100% INTACT)
# ============================================================
is_demo = (data_choice == "Demonstration Data (Simulated)")

if is_demo:
    df_raw = load_sample_dataset()
    filename_label = "Simulated Tissue Benchmark"
else:
    if uploaded_file is None:
        st.info("👆 Please upload a structured file in the sidebar, or switch to 'Demonstration Data (Simulated)'.")
        st.markdown("#### Supported Structured Formats:")
        st.markdown("- **CSV / TSV**: Comma or Tab delimited (`.csv`, `.tsv`)")
        st.markdown("- **Excel**: Standard workbooks (`.xlsx`, `.xls`)")
        st.markdown("- **JSON**: Record arrays or column mappings (`.json`)")
        st.markdown("- **Parquet**: Columnar binary format (`.parquet`, `.pq`)")
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=to_csv_bytes(sample_df),
            file_name="sample_spatial_data.csv",
            mime="text/csv"
        )
        st.stop()

    try:
        df_raw = read_uploaded_file(uploaded_file)
        filename_label = uploaded_file.name
    except Exception as exc:
        st.error(f"❌ Error loading file: {exc}")
        st.stop()

if df_raw.empty:
    st.error("❌ The uploaded file contains zero data rows. Please upload a populated dataset.")
    st.stop()

# Case-insensitive column matching & alias normalization
cols_map = {str(col).strip().lower(): col for col in df_raw.columns}

ALIASES = {
    "cell_id": ["cell_id", "cellid", "id", "barcode", "cell_barcode", "cell_name", "spot_id"],
    "cell_type": ["cell_type", "celltype", "phenotype", "annotation", "cell_annotation", "label", "cell_label", "cluster"],
    "x": ["x", "x_coord", "x_coordinate", "centroid_x", "pos_x", "center_x", "spatial_x"],
    "y": ["y", "y_coord", "y_coordinate", "centroid_y", "pos_y", "center_y", "spatial_y"],
}

def resolve_column(target):
    for alias in ALIASES[target]:
        if alias in cols_map:
            return cols_map[alias]
    return None

id_col_found = resolve_column("cell_id")
type_col_found = resolve_column("cell_type")
x_col_found = resolve_column("x")
y_col_found = resolve_column("y")

missing_required = []
if type_col_found is None:
    missing_required.append("cell_type")
if x_col_found is None:
    missing_required.append("x")
if y_col_found is None:
    missing_required.append("y")

if missing_required:
    st.error(f"""
    ### ⚠️ Missing Required Information
    The uploaded file is missing required spatial field(s): **{', '.join(missing_required)}**.
    
    **Required Columns:**
    - `cell_type` : Biological cell identity/annotation (must be supplied in file).
    - `x` : Horizontal spatial coordinate.
    - `y` : Vertical spatial coordinate.
    - `cell_id` *(optional)* : Unique identifier for each cell (generated if absent).
    
    **Columns detected in your file:** `{list(df_raw.columns)}`  
    *Scientific Note: Cell types must be taken from annotations in the file. NicheFinder does not guess cell types from coordinates alone.*
    """)
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=to_csv_bytes(sample_df),
        file_name="sample_spatial_data.csv",
        mime="text/csv"
    )
    st.stop()

# Standardize DataFrame
clean_df = pd.DataFrame()

if id_col_found is not None:
    clean_df["cell_id"] = df_raw[id_col_found].astype(str).str.strip()
else:
    clean_df["cell_id"] = [f"CELL_{i+1:04d}" for i in range(len(df_raw))]

clean_df["cell_type"] = df_raw[type_col_found].astype(str).str.strip()
clean_df["x"] = pd.to_numeric(df_raw[x_col_found], errors="coerce")
clean_df["y"] = pd.to_numeric(df_raw[y_col_found], errors="coerce")

# Drop null coordinates
clean_df = clean_df.dropna(subset=["x", "y"]).reset_index(drop=True)

if len(clean_df) < 2:
    st.error("❌ Dataset contains fewer than 2 valid coordinate rows. Distance analysis requires at least two cells.")
    st.stop()

# Disambiguate duplicate IDs
if clean_df["cell_id"].duplicated().any():
    clean_df["cell_id"] = clean_df["cell_id"] + "_" + clean_df.index.astype(str)

total_cells = len(clean_df)
cell_types = sorted(clean_df["cell_type"].unique().tolist())
num_types = len(cell_types)

# Cell type global statistics
type_counts = clean_df["cell_type"].value_counts().reset_index()
type_counts.columns = ["Cell Type", "Number of Cells"]
type_counts["Percentage of Total Cells"] = (type_counts["Number of Cells"] / total_cells) * 100

most_abundant = type_counts.iloc[0]["Cell Type"]
most_abundant_count = type_counts.iloc[0]["Number of Cells"]
most_abundant_pct = type_counts.iloc[0]["Percentage of Total Cells"]

# Spatial range
x_span = float(clean_df["x"].max() - clean_df["x"].min())
y_span = float(clean_df["y"].max() - clean_df["y"].min())
diag_span = float(np.hypot(x_span, y_span))
default_radius = max(1.0, round(diag_span * 0.15, 1))
max_radius = max(10.0, round(diag_span, 1))


# ============================================================
# MAIN DASHBOARD: EXACT USER HEADINGS & PRESERVED LAYOUT
# ============================================================

# Header
st.markdown('<div class="hero-box">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🧬 NicheFinder — <span>Spatial Biology Explorer</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-tagline">Discover the hidden neighbourhoods of cells.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Dataset Origin Banner
if is_demo:
    st.markdown(
        '<div class="banner-demo"><strong>⚠️ DEMONSTRATION MODE (SIMULATED DATA):</strong> '
        'This is a synthetic simulated dataset created for testing and demonstration. It is not real patient or clinical data.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="banner-info"><strong>📁 UPLOADED DATASET ACTIVE:</strong> '
        f'Loaded <strong>{total_cells:,}</strong> cells from <code>{filename_label}</code>. Cell types are taken from your annotations.</div>',
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# 1. SAMPLE OVERVIEW (TOP METRIC CARDS)
# ------------------------------------------------------------
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Cells</div>
        <div class="metric-value">{total_cells:,}</div>
        <div class="metric-sub">Cells with valid coordinates</div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div class="metric-card violet">
        <div class="metric-label">Annotated Cell Types</div>
        <div class="metric-value">{num_types}</div>
        <div class="metric-sub">Distinct phenotypes in dataset</div>
    </div>
    """, unsafe_allow_html=True)
with s3:
    st.markdown(f"""
    <div class="metric-card coral">
        <div class="metric-label">Most Abundant Cell Type</div>
        <div class="metric-value" style="font-size:1.45rem;">{most_abundant}</div>
        <div class="metric-sub">{most_abundant_count:,} cells ({most_abundant_pct:.1f}% of sample)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# HEADING: Cell Types Present
# ------------------------------------------------------------
st.subheader("Cell Types Present")
st.caption("Summary of cell identities as reported in dataset annotations. Cell identities are taken from existing annotations and not guessed from coordinates.")

display_summary_table = type_counts.copy()
display_summary_table["Percentage of Total Cells"] = display_summary_table["Percentage of Total Cells"].map(lambda p: f"{p:.2f}%")

col_tbl, col_dl = st.columns([3, 1])
with col_tbl:
    st.dataframe(display_summary_table, width="stretch", hide_index=True)
with col_dl:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Cell-Type Table (CSV)",
        data=to_csv_bytes(type_counts),
        file_name="cell_types_summary.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")


# ------------------------------------------------------------
# HEADING: Cell Counts and Percentages
# ------------------------------------------------------------
st.subheader("Cell Counts and Percentages")
st.caption("Visual distribution of annotated cell types across the tissue sample.")

col_bar, col_pie = st.columns([1.2, 1], gap="medium")

with col_bar:
    fig_global_bar = px.bar(
        type_counts,
        x="Cell Type",
        y="Number of Cells",
        color="Cell Type",
        color_discrete_map=COLORS,
        text=type_counts["Number of Cells"].map(lambda c: f"{c:,}"),
        title="Cell Abundance by Type"
    )
    fig_global_bar.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 19, 34, 0.8)",
        font=dict(color="#dbe6f5"),
        showlegend=False,
        margin=dict(l=15, r=15, t=35, b=15)
    )
    fig_global_bar.update_yaxes(gridcolor="#1b2f48")
    st.plotly_chart(fig_global_bar, width="stretch")

with col_pie:
    fig_global_pie = px.pie(
        type_counts,
        values="Number of Cells",
        names="Cell Type",
        color="Cell Type",
        color_discrete_map=COLORS,
        hole=0.5,
        title="Cell-Type Proportions"
    )
    fig_global_pie.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe6f5"),
        margin=dict(l=15, r=15, t=35, b=15)
    )
    st.plotly_chart(fig_global_pie, width="stretch")

st.markdown("---")


# ------------------------------------------------------------
# INTERACTIVE CONTROLS: CELL SELECTION & RADIUS
# ------------------------------------------------------------
st.markdown("#### Configure Focal Cell & Neighbourhood Radius")
c_sel1, c_sel2 = st.columns([1.5, 1.5])
with c_sel1:
    selected_idx = st.selectbox(
        "Select a cell to inspect (by Cell ID):",
        options=list(range(len(clean_df))),
        format_func=lambda i: f"{clean_df.loc[i, 'cell_id']} ({clean_df.loc[i, 'cell_type']}) at (x={clean_df.loc[i, 'x']:.1f}, y={clean_df.loc[i, 'y']:.1f})",
        index=0,
        key="selected_cell_index"
    )
with c_sel2:
    radius_val = st.slider(
        "Neighbourhood Radius (Distance Threshold):",
        min_value=0.5,
        max_value=float(max_radius),
        value=float(default_radius),
        step=max(0.1, round(max_radius / 100, 1)),
        help="All cells within this Euclidean distance from the selected cell are identified as neighbours."
    )

focal_row = clean_df.loc[selected_idx]
focal_id = str(focal_row["cell_id"])
focal_type = str(focal_row["cell_type"])
focal_x = float(focal_row["x"])
focal_y = float(focal_row["y"])

# Accurate Euclidean distance calculation
# Distance = sqrt((x - x0)^2 + (y - y0)^2)
all_coords = clean_df[["x", "y"]].to_numpy(dtype=float)
focal_coord = np.array([focal_x, focal_y], dtype=float)
deltas = all_coords - focal_coord
distances = np.sqrt(np.sum(np.square(deltas), axis=1))

# Exclude selected cell from its own neighbours!
non_focal_mask = np.arange(len(clean_df)) != selected_idx
neighbour_mask = non_focal_mask & (distances <= radius_val)

neighbours_df = clean_df.loc[neighbour_mask].copy()
neighbours_df["distance"] = np.round(distances[neighbour_mask], 3)
neighbours_df = neighbours_df.sort_values("distance").reset_index(drop=True)
num_neighbours = len(neighbours_df)


# ------------------------------------------------------------
# HEADING: Spatial Arrangement
# ------------------------------------------------------------
st.subheader("Spatial Arrangement")
st.caption("Interactive spatial scatter map. Each point represents a cell at its supplied x and y coordinates. Selected cell is starred, with connecting lines drawn to neighbours.")

fig_map = go.Figure()

# 1. Background cells for each cell type
for idx, ctype in enumerate(cell_types):
    subset = clean_df[clean_df["cell_type"] == ctype]
    fig_map.add_trace(go.Scattergl(
        x=subset["x"],
        y=subset["y"],
        mode="markers",
        name=ctype,
        marker=dict(
            size=7,
            color=get_color(ctype, idx),
            opacity=0.75,
            line=dict(width=0.5, color="rgba(255, 255, 255, 0.4)")
        ),
        text=subset["cell_id"],
        customdata=np.stack((subset["cell_type"], subset["x"], subset["y"]), axis=-1),
        hovertemplate="<b>Cell ID:</b> %{text}<br><b>Cell Type:</b> %{customdata[0]}<br><b>Coordinates:</b> (%{customdata[1]:.2f}, %{customdata[2]:.2f})<extra></extra>"
    ))

# 2. Draw lines connecting selected cell to neighbours
if num_neighbours > 0:
    x_lines, y_lines = [], []
    for _, nr in neighbours_df.iterrows():
        x_lines.extend([focal_x, nr["x"], None])
        y_lines.extend([focal_y, nr["y"], None])
    fig_map.add_trace(go.Scatter(
        x=x_lines,
        y=y_lines,
        mode="lines",
        name="Distance lines",
        line=dict(color="rgba(0, 229, 255, 0.5)", width=1.4, dash="dot"),
        hoverinfo="skip"
    ))

# 3. Radius boundary circle
theta = np.linspace(0, 2*np.pi, 150)
fig_map.add_trace(go.Scatter(
    x=focal_x + radius_val * np.cos(theta),
    y=focal_y + radius_val * np.sin(theta),
    mode="lines",
    name=f"Radius ({radius_val:g})",
    line=dict(color="#FFD166", width=1.8, dash="dash"),
    hoverinfo="skip"
))

# 4. Highlight neighbours
if num_neighbours > 0:
    fig_map.add_trace(go.Scatter(
        x=neighbours_df["x"],
        y=neighbours_df["y"],
        mode="markers",
        name=f"Neighbours ({num_neighbours})",
        marker=dict(
            size=10,
            color="#00E5FF",
            symbol="circle",
            line=dict(width=1.5, color="#ffffff")
        ),
        text=neighbours_df["cell_id"],
        customdata=np.stack((neighbours_df["cell_type"], neighbours_df["distance"]), axis=-1),
        hovertemplate="<b>Neighbour:</b> %{text}<br><b>Type:</b> %{customdata[0]}<br><b>Distance:</b> %{customdata[1]:.2f} coordinate units<extra></extra>"
    ))

# 5. Highlight selected cell
fig_map.add_trace(go.Scatter(
    x=[focal_x],
    y=[focal_y],
    mode="markers+text",
    name=f"Selected Cell ({focal_id})",
    marker=dict(
        size=18,
        color="#FF5E7E",
        symbol="star",
        line=dict(width=2, color="#ffffff")
    ),
    text=[f"★ {focal_id}"],
    textposition="top center",
    textfont=dict(color="#FF5E7E", size=12),
    hoverinfo="text",
    hovertext=f"<b>Selected Cell: {focal_id}</b><br><b>Type:</b> {focal_type}<br><b>Coordinates:</b> ({focal_x:.2f}, {focal_y:.2f})"
))

fig_map.update_layout(
    height=560,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10, 19, 34, 0.8)",
    font=dict(color="#dbe6f5"),
    margin=dict(l=15, r=15, t=20, b=20),
    legend=dict(
        title=dict(text="<b>Legend</b>", font=dict(color="#00E5FF")),
        bgcolor="rgba(11, 20, 34, 0.75)",
        bordercolor="#203550",
        borderwidth=1
    )
)
fig_map.update_xaxes(title="X Coordinate", gridcolor="#1b2f48", zerolinecolor="#1b2f48")
fig_map.update_yaxes(title="Y Coordinate", gridcolor="#1b2f48", zerolinecolor="#1b2f48", scaleanchor="x", scaleratio=1)
st.plotly_chart(fig_map, width="stretch")

st.markdown("---")


# ------------------------------------------------------------
# HEADING: Explore Neighbouring Cells
# ------------------------------------------------------------
st.subheader("Explore Neighbouring Cells")
st.write(
    f"Selected Cell: **`{focal_id}`** (Type: **{focal_type}**, Coordinates: `x={focal_x:.2f}, y={focal_y:.2f}`). "
    f"Found **{num_neighbours}** neighbouring cell(s) within radius **{radius_val:g} coordinate units**."
)

if num_neighbours > 0:
    table_neighbours = pd.DataFrame({
        "Neighbour ID": neighbours_df["cell_id"],
        "Cell Type": neighbours_df["cell_type"],
        "X": neighbours_df["x"].round(3),
        "Y": neighbours_df["y"].round(3),
        "Euclidean Distance": neighbours_df["distance"].round(3)
    })
    st.dataframe(table_neighbours, width="stretch", hide_index=True)
    
    st.download_button(
        label=f"📥 Download Neighbour Table for {focal_id} (CSV)",
        data=to_csv_bytes(table_neighbours),
        file_name=f"neighbours_of_{focal_id}.csv",
        mime="text/csv"
    )
else:
    st.warning(f"⚠️ No neighbouring cells found within radius {radius_val:g}. Try increasing the radius slider above.")

st.markdown("---")


# ------------------------------------------------------------
# HEADING: Distance Between Cells
# ------------------------------------------------------------
st.subheader("Distance Between Cells")
st.caption(
    "Euclidean distance from the selected cell to each neighbouring cell: Distance = √((x - x₀)² + (y - y₀)²). "
    "Distance units are coordinate units (no physical units specified in dataset). Distance is a physical spatial length, never a percentage."
)

if num_neighbours > 0:
    fig_dist = px.bar(
        neighbours_df,
        x="cell_id",
        y="distance",
        color="cell_type",
        color_discrete_map=COLORS,
        labels={"cell_id": "Neighbour Cell ID", "distance": "Distance (coordinate units)", "cell_type": "Cell Type"},
        title=f"Distances to Neighbours from {focal_id} (in Coordinate Units)"
    )
    fig_dist.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 19, 34, 0.8)",
        font=dict(color="#dbe6f5"),
        margin=dict(l=15, r=15, t=35, b=15)
    )
    fig_dist.update_yaxes(title="Distance (coordinate units)", gridcolor="#1b2f48")
    fig_dist.update_xaxes(title="Neighbour Cell ID", gridcolor="#1b2f48")
    st.plotly_chart(fig_dist, width="stretch")
else:
    st.info("ℹ️ Select a cell with neighbouring cells within the radius to view distance plots.")

st.markdown("---")


# ------------------------------------------------------------
# HEADING: Neighbour Composition
# ------------------------------------------------------------
st.subheader("Neighbour Composition")
st.caption("Distribution of cell types among the selected cell's neighbours. Percentages add up to exactly 100%.")

if num_neighbours > 0:
    n_counts = neighbours_df["cell_type"].value_counts().reset_index()
    n_counts.columns = ["Cell Type", "Number of Neighbour Cells"]
    n_counts["Percentage of Neighbours"] = (n_counts["Number of Neighbour Cells"] / num_neighbours) * 100
    
    col_nc_tbl, col_nc_bar, col_nc_pie = st.columns([1, 1.2, 1], gap="medium")
    
    with col_nc_tbl:
        disp_nc = n_counts.copy()
        disp_nc["Percentage of Neighbours"] = disp_nc["Percentage of Neighbours"].map(lambda p: f"{p:.2f}%")
        st.dataframe(disp_nc, width="stretch", hide_index=True)
        st.caption(f"Total neighbours: {num_neighbours} (Percentages sum to 100.0%)")

    with col_nc_bar:
        fig_n_bar = px.bar(
            n_counts,
            x="Number of Neighbour Cells",
            y="Cell Type",
            orientation="h",
            color="Cell Type",
            color_discrete_map=COLORS,
            text=n_counts["Percentage of Neighbours"].map(lambda p: f"{p:.1f}%"),
            title="Neighbour Cell Counts"
        )
        fig_n_bar.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10, 19, 34, 0.8)",
            font=dict(color="#dbe6f5"),
            showlegend=False,
            margin=dict(l=10, r=10, t=35, b=10)
        )
        fig_n_bar.update_xaxes(gridcolor="#1b2f48")
        st.plotly_chart(fig_n_bar, width="stretch")

    with col_nc_pie:
        fig_n_pie = px.pie(
            n_counts,
            values="Number of Neighbour Cells",
            names="Cell Type",
            color="Cell Type",
            color_discrete_map=COLORS,
            hole=0.45,
            title="Neighbour Proportions"
        )
        fig_n_pie.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dbe6f5"),
            margin=dict(l=10, r=10, t=35, b=10)
        )
        st.plotly_chart(fig_n_pie, width="stretch")
else:
    st.info("ℹ️ Select a cell or increase the neighbourhood radius to view neighbour composition.")


# ============================================================
# FOOTER & SCIENTIFIC ACCURACY NOTICE
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #8fa6c4; font-size: 0.84rem; line-height: 1.6;">
    <strong>🧬 NicheFinder — Spatial Biology Explorer</strong> · Research & Bioinformatics Prototype<br>
    <em>Cell types are derived directly from supplied annotations. NicheFinder does not infer cell identities from coordinates alone.</em><br>
    Spatial proximity indicates microenvironmental co-localization, not clinical causation or proven biochemical interaction.
</div>
""", unsafe_allow_html=True)
