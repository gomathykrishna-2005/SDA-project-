"""Top-K research + live MSANA dashboard.

Run from the SDA-project repository:
    streamlit run research_dashboard.py

The dashboard only plots research files and values actually returned by InfluxDB.
It does not generate placeholder metrics or experimental results.
"""
from pathlib import Path
import os
import time
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from influxdb_client import InfluxDBClient

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CSV_DIR = RESULTS / "csv"
PLOT_DIR = RESULTS / "plots"

st.set_page_config(
    page_title="Top-K Research Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme selection: Dark / Light ----------
# Keep the choice in Streamlit session state so it survives dashboard reruns.
if "dashboard_theme" not in st.session_state:
    st.session_state.dashboard_theme = "Dark"

theme = st.sidebar.radio(
    "🎨 Appearance",
    ["Dark", "Light"],
    horizontal=True,
    key="dashboard_theme",
    help="Switch the dashboard interface and Plotly charts between dark and light mode.",
)

if theme == "Dark":
    C = {
        "app": "#080d18", "sidebar": "#0d1525", "panel": "#101a2b",
        "border": "#243650", "text": "#e7edf8", "muted": "#9eb0ca",
        "grid": "#263650", "plot": "#0d1525", "hero1": "#111e35",
        "hero2": "#14243a", "accent": "#72b7ff", "template": "plotly_dark",
    }
else:
    C = {
        "app": "#f4f7fb", "sidebar": "#ffffff", "panel": "#ffffff",
        "border": "#dbe3ef", "text": "#172033", "muted": "#58677d",
        "grid": "#e2e8f0", "plot": "#ffffff", "hero1": "#ffffff",
        "hero2": "#eaf2ff", "accent": "#245edb", "template": "plotly_white",
    }

st.markdown(f"""
<style>
:root {{ color-scheme: {'dark' if theme == 'Dark' else 'light'}; }}
.stApp {{ background: {C['app']}; color: {C['text']}; }}
[data-testid="stHeader"] {{ background: {C['app']}; }}
[data-testid="stSidebar"] {{ background: {C['sidebar']}; border-right: 1px solid {C['border']}; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1600px; }}
.hero {{ padding: 1.4rem 1.6rem; border: 1px solid {C['border']}; border-radius: 18px;
        background: linear-gradient(120deg,{C['hero1']} 0%,{C['panel']} 58%,{C['hero2']} 100%); margin-bottom: 1.2rem; }}
.hero h1 {{ margin: 0; color: {C['text']}; font-size: 2.1rem; letter-spacing: -.03em; }}
.hero p {{ margin: .45rem 0 0 0; color: {C['muted']}; font-size: 1rem; }}
.section-label {{ color:{C['accent']}; text-transform:uppercase; letter-spacing:.13em; font-size:.76rem; font-weight:750; margin:.6rem 0 .45rem; }}
div[data-testid="stMetric"] {{ background:{C['panel']}; border:1px solid {C['border']}; padding:1rem; border-radius:14px; }}
div[data-testid="stMetricLabel"] {{ color:{C['muted']}; }}
div[data-testid="stMetricValue"] {{ color:{C['text']}; }}
div[data-testid="stAlert"] {{ border-radius:12px; }}
[data-testid="stPlotlyChart"] {{ background:{C['plot']}; border:1px solid {C['border']}; border-radius:14px; padding:.35rem; }}
hr {{ border-color:{C['border']}; }}
.small-note {{ color:{C['muted']}; font-size:.84rem; }}
/* Keep common form controls legible in both themes. */
[data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stTextInput"] {{ color:{C['text']}; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template=C["template"],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="Inter, Arial, sans-serif"),
    margin=dict(l=24, r=20, t=55, b=35),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)


def style_fig(fig, height=360):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(showgrid=True, gridcolor=C["grid"], zerolinecolor=C["border"], color=C["muted"])
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"], zerolinecolor=C["border"], color=C["muted"])
    return fig


def discover_csvs():
    if not CSV_DIR.exists():
        return []
    return sorted(CSV_DIR.rglob("*.csv"))


def discover_plots(dataset):
    if not PLOT_DIR.exists():
        return []
    files = sorted([p for p in PLOT_DIR.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    # Prefer plots with the selected dataset in the filename; otherwise show all available.
    matched = [p for p in files if dataset.lower() in p.stem.lower()]
    return matched if matched else files


def read_csv_safely(path):
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return None


def load_research_data(dataset):
    """Load existing CSV files; no data is fabricated if a file is absent."""
    frames = []
    for path in discover_csvs():
        df = read_csv_safely(path)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["_source_file"] = path.name
        # Match dataset from filename if named; otherwise keep generic experiment files.
        named_datasets = ["CICIDS2017", "IoTID20", "Synthetic"]
        found = next((d for d in named_datasets if d.lower() in path.stem.lower()), None)
        if found is None or found.lower() == dataset.lower():
            frames.append(df)
    return frames


def find_time_column(df):
    for col in df.columns:
        if str(col).lower() in {"time", "timestamp", "datetime", "_time", "date"}:
            return col
    return None


def query_influx(url, org, bucket, token, hours):
    """Query raw points for the selected period, keeping actual Influx tags/fields."""
    if not token:
        return pd.DataFrame(), "Add your InfluxDB token in the sidebar or set INFLUX_TOKEN."
    flux = f'''from(bucket: "{bucket}")
  |> range(start: -{int(hours)}h)
  |> keep(columns: ["_time", "_measurement", "_field", "_value"])
  |> sort(columns: ["_time"], desc: false)
  |> limit(n: 10000)'''
    try:
        with InfluxDBClient(url=url, token=token, org=org, timeout=12000) as client:
            result = client.query_api().query_data_frame(query=flux, org=org)
        if isinstance(result, list):
            parts = [part for part in result if isinstance(part, pd.DataFrame) and not part.empty]
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        elif isinstance(result, pd.DataFrame):
            df = result.copy()
        else:
            df = pd.DataFrame()
        if df.empty:
            return df, None
        keep = [c for c in ["_time", "_measurement", "_field", "_value"] if c in df.columns]
        df = df[keep].copy()
        if "_time" in df:
            df["_time"] = pd.to_datetime(df["_time"], errors="coerce", utc=True)
        if "_value" in df:
            df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
        df = df.dropna(subset=["_time", "_value"])
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"InfluxDB query failed: {type(exc).__name__}: {exc}"


# ---------- Sidebar ----------
st.sidebar.markdown("## ⚙️ Dashboard settings")
dataset = st.sidebar.selectbox("Research dataset", ["CICIDS2017", "IoTID20", "Synthetic"])
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔌 InfluxDB connection")
influx_url = st.sidebar.text_input("InfluxDB URL", value=os.getenv("INFLUX_URL", "http://localhost:8086"))
influx_org = st.sidebar.text_input("Organization", value=os.getenv("INFLUX_ORG", "SDA"))
influx_bucket = st.sidebar.text_input("Bucket", value=os.getenv("INFLUX_BUCKET", "MSANA"))
influx_token = st.sidebar.text_input("InfluxDB token", value=os.getenv("INFLUX_TOKEN", ""), type="password", help="Token stays in this local app session. Do not commit it to Git.")
hours = st.sidebar.selectbox("Live-data time window", [1, 3, 6, 12, 24, 48], index=4, format_func=lambda h: f"Last {h} hour(s)")
auto_refresh = st.sidebar.toggle("Auto-refresh live graphs", value=False)
refresh_seconds = st.sidebar.selectbox("Refresh interval", [10, 15, 30, 60], index=2, format_func=lambda x: f"Every {x} seconds", disabled=not auto_refresh)
refresh_now = st.sidebar.button("↻ Refresh dashboard", use_container_width=True)

if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_seconds * 1000, key="live_dashboard_refresh")
    except ImportError:
        st.sidebar.warning("For automatic refresh, install streamlit-autorefresh. Manual refresh still works.")

# Avoid stale values between explicit refreshes; short cache only when auto refresh is off.
cache_key = f"{influx_url}|{influx_org}|{influx_bucket}|{hours}|{bool(influx_token)}"
if refresh_now or auto_refresh or "last_cache_key" not in st.session_state or st.session_state.last_cache_key != cache_key:
    st.session_state.pop("influx_cache", None)
    st.session_state.last_cache_key = cache_key

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>📡 Top-K Stream Selection</h1>
  <p>Multistream Concept Drift Adaptation · Research Evaluation &amp; Live MSANA Monitoring</p>
</div>
""", unsafe_allow_html=True)

research_tab, live_tab, files_tab = st.tabs(["🧪 Research results", "🟢 Live MSANA", "📁 Data & provenance"])

# ---------- Research tab ----------
with research_tab:
    st.markdown('<div class="section-label">Research experiment outputs</div>', unsafe_allow_html=True)
    plot_files = discover_plots(dataset)
    csv_frames = load_research_data(dataset)
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset selected", dataset)
    c2.metric("Matching plot files", len(plot_files))
    c3.metric("CSV files loaded", len(csv_frames))
    st.caption("These panels use files already present in results/plots and results/csv. Missing outputs are not replaced with sample results.")

    if plot_files:
        # Show all discovered plots in a tidy two-column gallery.
        for start in range(0, len(plot_files), 2):
            cols = st.columns(2)
            for offset, col in enumerate(cols):
                idx = start + offset
                if idx >= len(plot_files):
                    continue
                path = plot_files[idx]
                with col:
                    st.markdown(f"**{path.stem.replace('_', ' ')}**")
                    st.image(str(path), use_container_width=True)
                    st.caption(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name)
    else:
        st.info(f"No image plots found for {dataset} in {PLOT_DIR}. Add or generate actual experiment plots to display them here.")

    st.markdown("---")
    st.markdown("### Explore existing experiment CSVs")
    if csv_frames:
        names = [f"{i+1}. {df['_source_file'].iloc[0]}" for i, df in enumerate(csv_frames)]
        selected_i = st.selectbox("CSV file", range(len(csv_frames)), format_func=lambda i: names[i])
        df = csv_frames[selected_i].drop(columns=["_source_file"], errors="ignore").copy()
        st.caption(f"Rows: {len(df):,} · Columns: {len(df.columns)}")
        st.dataframe(df.head(500), use_container_width=True, height=300)
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if df[c].notna().any()]
        if numeric_cols:
            x_options = ["Row index"] + list(df.columns)
            x_col = st.selectbox("X-axis", x_options, index=0, key="csv_x_axis")
            y_cols = st.multiselect("Numeric series to plot", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
            chart_df = df.copy().reset_index(names="Row index")
            if y_cols:
                if x_col == "Row index":
                    fig = px.line(chart_df, x="Row index", y=y_cols, markers=True, title="Existing CSV values")
                elif x_col in chart_df.columns:
                    fig = px.line(chart_df, x=x_col, y=y_cols, markers=True, title="Existing CSV values")
                st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        else:
            st.info("This CSV has no numeric columns to chart. Its rows are still available above.")
    else:
        st.info(f"No readable CSV files found under {CSV_DIR}.")

# ---------- Live monitoring tab ----------
with live_tab:
    st.markdown('<div class="section-label">Live data returned by InfluxDB</div>', unsafe_allow_html=True)
    st.caption(f"Source: {influx_url} · org={influx_org} · bucket={influx_bucket} · window=last {hours}h")
    # Query each rerun so the panel reflects actual bucket data.
    live_df, influx_error = query_influx(influx_url, influx_org, influx_bucket, influx_token, hours)

    if influx_error:
        st.error(influx_error)
    elif live_df.empty:
        st.warning("No numeric points were returned for this time window. InfluxDB may still contain data in other fields or older time ranges; try a wider window and verify the bucket and write pipeline.")
        st.markdown("**Next checks**")
        st.code('''from(bucket: "MSANA")
  |> range(start: -24h)
  |> keep(columns: ["_time", "_measurement", "_field", "_value"])
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 30)''', language="javascript")
    else:
        latest = live_df["_time"].max()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Points returned", f"{len(live_df):,}")
        metric_cols[1].metric("Measurements", live_df["_measurement"].nunique() if "_measurement" in live_df else "—")
        metric_cols[2].metric("Distinct fields", live_df["_field"].nunique() if "_field" in live_df else "—")
        metric_cols[3].metric("Latest point (UTC)", latest.strftime("%H:%M:%S") if pd.notna(latest) else "—")

        available_fields = sorted(live_df["_field"].dropna().astype(str).unique().tolist())
        selected_fields = st.multiselect("Fields to graph", available_fields, default=available_fields[:min(4, len(available_fields))])
        filtered = live_df[live_df["_field"].astype(str).isin(selected_fields)].copy() if selected_fields else live_df.iloc[0:0].copy()

        if not filtered.empty:
            # Each field can have a different scale; the field selector lets users keep charts readable.
            fig = px.line(
                filtered.sort_values("_time"), x="_time", y="_value", color="_field",
                line_group="_measurement", markers=True,
                hover_data={"_measurement": True, "_field": True, "_value": ":.6g", "_time": True},
                title="Live MSANA field values over time",
            )
            fig.update_layout(yaxis_title="Stored numeric value", xaxis_title="Time (UTC)")
            st.plotly_chart(style_fig(fig, 440), use_container_width=True)

            left, right = st.columns([1.15, 1])
            with left:
                st.markdown("### Latest values by field")
                latest_by_field = (filtered.sort_values("_time").groupby(["_measurement", "_field"], as_index=False).tail(1)
                                   .sort_values("_time", ascending=False))
                st.dataframe(latest_by_field[[c for c in ["_time", "_measurement", "_field", "_value"] if c in latest_by_field.columns]], use_container_width=True, height=300)
            with right:
                st.markdown("### Field distribution")
                dist = filtered.groupby("_field", as_index=False)["_value"].agg(["count", "mean", "min", "max"]).reset_index()
                if not dist.empty:
                    dist = dist.rename(columns={"count": "points", "mean": "mean", "min": "min", "max": "max"})
                    fig2 = px.bar(dist, x="_field", y="points", text="points", title="Points returned per field")
                    fig2.update_layout(xaxis_title="Field", yaxis_title="Point count")
                    st.plotly_chart(style_fig(fig2, 300), use_container_width=True)

        st.markdown("### Raw InfluxDB records")
        st.dataframe(live_df.sort_values("_time", ascending=False).head(500), use_container_width=True, height=260)
        st.download_button("⬇ Download queried points as CSV", live_df.to_csv(index=False).encode("utf-8"), file_name="influx_live_points.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("**Metric note:** accuracy, precision, recall, and F1 are shown only if those exact fields exist in the selected InfluxDB time range. Network event fields such as `ACK Flag Count` are graphed as event data, not relabelled as model-performance metrics.")

# ---------- Provenance tab ----------
with files_tab:
    st.markdown('<div class="section-label">Data inventory and reproducibility</div>', unsafe_allow_html=True)
    st.write("The dashboard reads local experiment artifacts; it does not rerun training or modify the Top-K implementation.")
    st.markdown("### Research CSV inventory")
    csv_paths = discover_csvs()
    if csv_paths:
        inv = []
        for p in csv_paths:
            inv.append({"File": str(p.relative_to(ROOT)), "Size (KB)": round(p.stat().st_size / 1024, 1), "Modified": pd.to_datetime(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")})
        st.dataframe(pd.DataFrame(inv), use_container_width=True)
    else:
        st.info(f"No CSVs found in {CSV_DIR}.")
    st.markdown("### Plot inventory")
    all_plots = sorted([p for p in PLOT_DIR.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]) if PLOT_DIR.exists() else []
    if all_plots:
        st.dataframe(pd.DataFrame([{"File": str(p.relative_to(ROOT)), "Size (KB)": round(p.stat().st_size / 1024, 1)} for p in all_plots]), use_container_width=True)
    else:
        st.info(f"No plot images found in {PLOT_DIR}.")
    st.markdown("### InfluxDB schema check")
    st.write("Use the Live MSANA tab to see actual measurement names, field names, timestamps, and numeric values returned by the query. If the bucket contains only network-event fields, the dashboard will not claim that model metrics are present.")

st.markdown("---")
st.markdown('<p class="small-note">Top-K Stream Selection · Local research dashboard · Experimental claims must be supported by saved results and actual InfluxDB records.</p>', unsafe_allow_html=True)
