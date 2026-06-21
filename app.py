"""
ASTraM — Adaptive Smart Traffic and Resource Management
Streamlit deployment app.

Loads artefacts produced by ASTraM_Final_Fixed.ipynb (models/ directory).
Run with:  streamlit run app.py
"""

import os
import json
import warnings
from datetime import datetime
 
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
 
warnings.filterwarnings("ignore")
 
import streamlit as st
 
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
 
import xgboost as xgb

# Page config
st.set_page_config(
    page_title="ASTraM — Traffic Intelligence",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background:#f4f6f9;border:1px solid #dde2ea;
    border-radius:10px;padding:14px 18px;
}
button[data-baseweb="tab"] { font-size:15px;font-weight:500; }
.section-hdr {
    font-size:16px;font-weight:600;
    border-left:4px solid #3a86ff;
    padding-left:10px;margin:18px 0 10px;
}
</style>
""", unsafe_allow_html=True)

# Constants
LOG_FILE    = "event_feedback_log.json"
MODELS_DIR  = "models"
 
# Update this to match your CSV filename
CSV_PATH = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
 
NODE_COORDS = {
    "CBD 1":                  [12.9716, 77.5946],
    "CBD 2":                  [12.9650, 77.5900],
    "Hosur Road":             [12.9244, 77.6217],
    "Mysore Road":            [12.9387, 77.5353],
    "Magadi Road":            [12.9750, 77.5350],
    "Tumkur Road":            [13.0334, 77.5381],
    "Bellary Road 1":         [13.0400, 77.5900],
    "Bellary Road 2":         [13.0800, 77.5950],
    "Airport New South Road": [13.1000, 77.6500],
    "Old Madras Road":        [12.9900, 77.6500],
    "Old Airport Road":       [12.9550, 77.6500],
    "Varthur Road":           [12.9500, 77.7000],
    "Bannerghata Road":       [12.8900, 77.5950],
    "West of Chord Road":     [12.9900, 77.5500],
    "ORR West 1":             [12.9800, 77.5400],
    "ORR North 1":            [13.0250, 77.6340],
    "ORR North 2":            [13.0100, 77.6500],
    "ORR East 1":             [12.9400, 77.6800],
    "ORR East 2":             [12.9200, 77.6700],
    "Hennur Main Road":       [13.0300, 77.6300],
    "IRR(Thanisandra road)":  [13.0500, 77.6300],
}
 
BASE_EDGES = [
    ("CBD 1","CBD 2",5), ("CBD 1","Old Airport Road",15), ("CBD 1","Hosur Road",20),
    ("CBD 1","Bellary Road 1",18), ("CBD 1","Magadi Road",15), ("CBD 1","Old Madras Road",16),
    ("Old Airport Road","Varthur Road",12), ("Old Airport Road","ORR East 1",10),
    ("Hosur Road","ORR East 2",15), ("Hosur Road","Bannerghata Road",18),
    ("ORR East 1","ORR East 2",10), ("ORR East 1","Old Madras Road",12),
    ("Bellary Road 1","Bellary Road 2",10), ("Bellary Road 2","Airport New South Road",15),
    ("Bellary Road 1","ORR North 1",12), ("ORR North 1","ORR North 2",8),
    ("ORR North 2","Hennur Main Road",10), ("Hennur Main Road","IRR(Thanisandra road)",8),
    ("Magadi Road","West of Chord Road",10), ("West of Chord Road","Tumkur Road",12),
    ("Tumkur Road","ORR West 1",10), ("ORR West 1","Mysore Road",15),
    ("Mysore Road","Bannerghata Road",25),
]
 
 
CORRIDOR_ZONE_MAP = {
    "CBD 1":"Central","CBD 2":"Central","Hosur Road":"South Zone 1",
    "Mysore Road":"West Zone 2","Magadi Road":"West Zone 1",
    "Tumkur Road":"North West Zone","Bellary Road 1":"North Zone 1",
    "Bellary Road 2":"North Zone 1","Airport New South Road":"North Zone 2",
    "Old Madras Road":"East Zone 1","Old Airport Road":"East Zone 1",
    "Varthur Road":"East Zone 2","Bannerghata Road":"South Zone 2",
    "West of Chord Road":"West Zone 1","ORR West 1":"West Zone 2",
    "ORR North 1":"North Zone 2","ORR North 2":"North Zone 2",
    "ORR East 1":"East Zone 1","ORR East 2":"East Zone 1",
    "Hennur Main Road":"North Zone 2","IRR(Thanisandra road)":"North Zone 2",
}
 
CAUSE_OPTIONS = [
    "vehicle_breakdown","tree_fall","accident","congestion",
    "water_logging","construction","public_event","procession","vip_movement",
]
CORRIDOR_LIST = list(NODE_COORDS.keys())
DAY_NAMES     = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# Load models
@st.cache_resource
def load_models():
    m = {}
    files = {
        "severity_clf":    ("severity_xgb.json",             "xgb"),
        "delay_reg":       ("delay_xgb.json",                "xgb"),
        "radius_reg":      ("radius_xgb.json",               "xgb"),
        "label_encoders":  ("label_encoders.pkl",            "pkl"),
        "feature_cols":    ("feature_cols.pkl",              "pkl"),
        "sev_le":          ("severity_label_encoder.pkl",    "pkl"),
        "sev_labels":      ("sev_labels.pkl",                "pkl"),
        "hotspots":        ("hotspots.pkl",                  "pkl"),
        "shap_explainer":  ("shap_explainer.pkl",            "pkl"),
        "p90_caps":        ("duration_p90_caps.pkl",         "pkl"),
    }
    missing = []
    for key, (fname, ftype) in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        if ftype == "pkl":
            m[key] = joblib.load(path)
        else:
            if key.endswith("_clf"):
                obj = xgb.XGBClassifier()
            else:
                obj = xgb.XGBRegressor()
            obj.load_model(path)
            m[key] = obj
    return m, missing
 
models, missing_files = load_models()
 
@st.cache_data
def load_dataset():
    """Load the raw CSV and compute corridor-level statistics from real data."""
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["event_cause"] = df["event_cause"].str.lower().str.strip().fillna("others")
    df["corridor"]    = df["corridor"].fillna("Non-corridor")
    df["priority"]    = df["priority"].fillna("Low")
    df["requires_road_closure"] = df["requires_road_closure"].fillna(0).astype(int)
    df["start_dt"]    = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["hour"]        = df["start_dt"].dt.hour
    df["day_of_week"] = df["start_dt"].dt.dayofweek
    return df
 
df_raw = load_dataset()

# Graph helpers
@st.cache_resource
def build_graph():
    G = nx.Graph()
    G.add_weighted_edges_from(BASE_EDGES)
    return G
 
G_BASE = build_graph()
 
def probabilistic_graph(corridor_probs: dict) -> nx.Graph:
    G = G_BASE.copy()
    for u, v, data in G.edges(data=True):
        p = max(corridor_probs.get(u, 0.0), corridor_probs.get(v, 0.0))
        G[u][v]["weight"] = data["weight"] * (1.0 + p ** 2)
    return G
 
def best_route(G_dyn, start, end, blocked=None):
    G_tmp = G_dyn.copy()
    if blocked and G_tmp.has_node(blocked) and blocked not in (start, end):
        G_tmp.remove_node(blocked)
    try:
        path = nx.shortest_path(G_tmp, start, end, weight="weight")
        time = nx.shortest_path_length(G_tmp, start, end, weight="weight")
        return path, round(time, 1)
    except nx.NetworkXNoPath:
        return [], float("inf")

# Inference helpers
HOTSPOTS = models.get("hotspots", {
    "silk_board":(12.9176,77.6233),"mg_road":(12.9757,77.6011),
    "hebbal":(13.0358,77.5970),"whitefield":(12.9698,77.7499),
    "electronic_city":(12.8456,77.6603),"city_center":(12.9716,77.5946),
})
 
def safe_enc(le, val, fallback=0):
    try:
        return int(le.transform([str(val)])[0])
    except (ValueError, AttributeError):
        return fallback
 
def hotspot_distances(lat, lon):
    if lat is None or lon is None or np.isnan(float(lat)):
        return {f"dist_{k}_km": 0.0 for k in HOTSPOTS}
    return {f"dist_{k}_km": geodesic((lat, lon), coords).km
            for k, coords in HOTSPOTS.items()}

def build_row(cause, corridor, priority, hour, day, month,
              zone, veh_type, event_type, requires_closure,
              lat, lon, duration_hrs):
    le   = models.get("label_encoders", {})
    p90  = models.get("p90_caps", {})
    h    = int(hour)
    dow  = int(day)
    corr_zone = corridor + " | " + zone
 
    # Duration: apply P90 cap then log
    cap      = p90.get(cause, 48)
    dur_log  = float(np.log1p(min(duration_hrs, cap)))
 
    row = {
        "event_type_enc":        safe_enc(le.get("event_type"),     event_type),
        "event_cause_enc":       safe_enc(le.get("event_cause"),    cause),
        "priority_enc":          safe_enc(le.get("priority"),       priority),
        "corridor_enc":          safe_enc(le.get("corridor"),       corridor),
        "zone_enc":              safe_enc(le.get("zone"),           zone),
        "veh_type_enc":          safe_enc(le.get("veh_type"),       veh_type),
        "corridor_zone_enc":     safe_enc(le.get("corridor_zone"),  corr_zone),
        "requires_road_closure": int(requires_closure),
        "is_corridor":           int(corridor != "Non-corridor"),
        "is_weekend":            int(dow >= 5),
        "is_morning_peak":       int(7 <= h < 10),
        "is_evening_peak":       int(16 <= h < 21),
        "is_night":              int(h >= 22 or h < 6),
        "hour":                  h,
        "hour_sin":              float(np.sin(2 * np.pi * h / 24)),
        "hour_cos":              float(np.cos(2 * np.pi * h / 24)),
        "day_of_week":           dow,
        "month":                 int(month),
        "latitude":              float(lat) if lat else 12.97,
        "longitude":             float(lon) if lon else 77.59,
        "geo_cluster":           0,
        "route_length_km":       0.0,
        "duration_log":          dur_log,
        **hotspot_distances(lat, lon),
    }
 
    feature_cols = models.get("feature_cols", list(row.keys()))
    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]
 
def run_inference(input_df):
    clf    = models.get("severity_clf")
    r_del  = models.get("delay_reg")
    r_rad  = models.get("radius_reg")
    sev_le = models.get("sev_le")
    slabels= models.get("sev_labels", {0:"Low",1:"Moderate",2:"High",3:"Critical"})
 
    sev_enc  = int(clf.predict(input_df)[0])
    sev_prob = clf.predict_proba(input_df)[0]
    severity = slabels.get(sev_enc, "Unknown")
    delay    = float(r_del.predict(input_df)[0])
    radius   = float(r_rad.predict(input_df)[0])
    conf     = float(sev_prob[sev_enc])
 
    # High+Critical combined probability (for threshold at 0.30)
    high_idxs = [i for i, l in slabels.items() if l in ("High","Critical")]
    prob_high  = float(sum(sev_prob[i] for i in high_idxs))
 
    return severity, conf, delay, radius, prob_high, sev_prob

# Resource calculation
SEVERITY_MULT = {"Low":0.6, "Moderate":1.0, "High":1.5, "Critical":2.2}
 
def compute_resources(severity, duration_hrs, radius_km):
    mult       = SEVERITY_MULT.get(severity, 1.0)
    personnel  = max(4, int(duration_hrs * mult * 2.5))
    barricades = max(2, int(radius_km * 9 * mult))
    tow        = 0 if severity == "Low" else (1 if severity == "Moderate" else 2)
    return personnel, barricades, tow

# Diversion plans
DIVERSION = {
    "ORR": [
        "Close 1 inner lane; maintain at least 2 lanes for through traffic.",
        "Divert via Intermediate Ring Road or parallel service roads.",
        "Deploy personnel at upstream junctions 500 m, 1 km, and 2 km ahead.",
        "Activate variable message signs 2 km before the event location.",
    ],
    "CBD": [
        "Implement one-way contraflow on the nearest parallel street.",
        "Divert via Residency Road or Queens Road corridor.",
        "Close parking access on the affected side for the event duration.",
        "Coordinate with BMTC for bus route detours via MG Road.",
    ],
    "Tumkur": [
        "Divert cargo traffic via Peenya to Yeshwanthpur to Chord Road.",
        "Station a heavy-vehicle checkpoint 1 km before the event.",
        "Coordinate with NHAI for NH44 alternate routing if needed.",
        "Pre-position tow trucks at Peenya Industrial junction.",
    ],
    "Mysore": [
        "Divert via Kanakapura Road or Bannerghatta Road.",
        "Block all U-turns within 1 km of the event location.",
        "Station constables at the Mysore Road to Nice Road interchange.",
        "Alert KSRTC for intercity bus diversions if duration exceeds 2 hours.",
    ],
    "Hosur": [
        "Restrict entry from Bommanahalli junction towards the event site.",
        "Divert light vehicles via Sarjapur Road.",
        "Coordinate with Electronics City traffic post for signal timing changes.",
        "Deploy one tow truck on standby at Electronic City flyover.",
    ],
    "Bellary": [
        "Coordinate with airport security for Devanahalli-bound traffic advisory.",
        "Divert via Kogilu Cross to Hebbal flyover.",
        "Implement staggered signal cycles at Hebbal interchange.",
        "Deploy additional personnel at Esteem Mall junction.",
    ],
    "default": [
        "Station traffic personnel at the nearest major junction.",
        "Implement diversion via the next parallel road.",
        "Notify all affected BMTC routes for detour activation.",
        "Coordinate with the local police station for barricade deployment.",
    ],
}
 
N_STEPS = {"Low":1, "Moderate":2, "High":3, "Critical":4}
 
def diversion_plan(corridor, severity):
    n = N_STEPS.get(severity, 2)
    for key, steps in DIVERSION.items():
        if key.lower() in corridor.lower():
            return steps[:n]
    return DIVERSION["default"][:n]

# Feedback log
def append_log(record):
    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log = json.load(f)
    log.append(record)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
 
def load_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    with open(LOG_FILE) as f:
        return pd.DataFrame(json.load(f))

# Session state
for k, v in {"sim_run":False,"sim_result":None,"fb_done":False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Header
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a,#1b2838,#243447);
    padding:26px 32px;border-radius:12px;margin-bottom:22px;">
  <h1 style="color:#ffffff;margin:0;font-size:1.9rem;font-weight:700;letter-spacing:-.5px;">
    ASTraM
  </h1>
  <p style="color:#8aafd4;margin:6px 0 0;font-size:1rem;">
    Adaptive Smart Traffic and Resource Management &nbsp;·&nbsp; Gridlock Hackathon 2.0
  </p>
</div>
""", unsafe_allow_html=True)
 
if missing_files:
    st.warning(f"Model files not found: {', '.join(missing_files)}. "
               "Run ASTraM_Final.ipynb first to generate the models/ directory.")
if "severity_clf" not in models:
    st.error("Severity model not loaded. Cannot run simulation.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Journey")
    start_pt = st.selectbox("Start corridor", CORRIDOR_LIST, index=0)
    end_pt   = st.selectbox("Destination",    CORRIDOR_LIST, index=11)
 
    st.markdown("### Event")
    cause     = st.selectbox("Event cause",     CAUSE_OPTIONS)
    corridor  = st.selectbox("Event corridor",  CORRIDOR_LIST, index=10)
    priority  = st.selectbox("Priority",        ["High","Low"])
    event_type= st.selectbox("Event type",      ["unplanned","planned"])
    req_close = st.checkbox("Road closure reported", value=False)
 
    st.markdown("### Time")
    hour  = st.slider("Hour (0–23)",              0, 23, 18)
    day   = st.slider(f"Day of week (0=Monday)",  0,  6,  2)
    month = st.slider("Month",                    1, 12,  6)
    dur   = st.slider("Expected duration (hrs)",  0.5, 8.0, 2.0, 0.5)
 
    st.caption(
        f"{DAY_NAMES[day]}  "
        f"{'— Morning peak' if 7<=hour<10 else '— Evening peak' if 16<=hour<21 else '— Off-peak'}"
    )
 
    st.markdown("### Location (optional)")
    lat_in = st.number_input("Latitude",  value=float(NODE_COORDS[corridor][0]), format="%.4f")
    lon_in = st.number_input("Longitude", value=float(NODE_COORDS[corridor][1]), format="%.4f")
 
    veh_type = st.selectbox("Vehicle type", ["none","car","bmtc_bus","truck","two_wheeler"])
    zone      = CORRIDOR_ZONE_MAP.get(corridor, "Unknown")
    st.caption(f"Zone: {zone}")
 
    st.markdown("---")
    if st.button("Run Simulation", type="primary", use_container_width=True):
        st.session_state.sim_run    = True
        st.session_state.sim_result = None
        st.session_state.fb_done    = False

# Tabs
tab_sim, tab_impact, tab_shap, tab_log = st.tabs([
    "Live Simulation",
    "Impact Dashboard",
    "AI Explainability",
    "Feedback and Learning",
])
 

# TAB 1 — SIMULATION
with tab_sim:
    if not st.session_state.sim_run:
        st.info("Configure the event in the sidebar and click Run Simulation.")
        m0 = folium.Map(location=[12.97, 77.59], zoom_start=12, tiles="CartoDB positron")
        for node, coord in NODE_COORDS.items():
            folium.CircleMarker(coord, radius=4, color="#3a86ff",
                                fill=True, fill_opacity=0.5, tooltip=node).add_to(m0)
        for u, v, _ in BASE_EDGES:
            folium.PolyLine([NODE_COORDS[u],NODE_COORDS[v]],
                            color="#cccccc",weight=1.5,opacity=0.4).add_to(m0)
        st_folium(m0, width="100%", height=440)
 
    else:
        # ── Inference ─────────────────────────────────────────────────
        inp_df = build_row(
            cause, corridor, priority, hour, day, month,
            zone, veh_type, event_type, req_close, lat_in, lon_in, dur
        )
        severity, conf, delay, radius, prob_high, sev_prob = run_inference(inp_df)
 
        # Use 0.30 threshold for High/Critical alert
        alert_high = prob_high >= 0.30
 
        # Resources
        personnel, barricades, tow = compute_resources(severity, dur, radius)
 
        # Graph routing
        prob_dict  = {corridor: min(prob_high, 1.0)}
        G_dyn      = probabilistic_graph(prob_dict)
        blocked    = corridor if (severity in ("High","Critical") or req_close) else None
        path, eta  = best_route(G_dyn, start_pt, end_pt, blocked=blocked)
        _, norm_t  = best_route(G_BASE, start_pt, end_pt)
        delay_add  = max(0, round(eta - norm_t, 1))
 
        # Diversion
        div_steps = diversion_plan(corridor, severity)
 
        # Save to session
        st.session_state.sim_result = dict(
            severity=severity, conf=conf, delay=delay, radius=radius,
            prob_high=prob_high, alert_high=alert_high, sev_prob=sev_prob,
            personnel=personnel, barricades=barricades, tow=tow,
            path=path, eta=eta, norm_t=norm_t, delay_add=delay_add,
            div_steps=div_steps, prob_dict=prob_dict, blocked=blocked,
            inp_df=inp_df,
        )
        r = st.session_state.sim_result
 
        # ── KPI row ────────────────────────────────────────────────────
        SEV_COLORS = {"Low":"#2dc653","Moderate":"#ffd166","High":"#ef476f","Critical":"#6a0572"}
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Severity",        r["severity"])
        k2.metric("High/Critical prob (thresh=0.30)",
                  f"{r['prob_high']:.0%}",
                  delta="ALERT" if r["alert_high"] else "Normal",
                  delta_color="inverse")
        k3.metric("Predicted delay",  f"{r['delay']:.0f} min")
        k4.metric("Officers needed",  r["personnel"])
 
        st.markdown("---")
        left, right = st.columns([1, 2])
 
        # ── Left panel ──────────────────────────────────────────────────
        with left:
            st.markdown('<div class="section-hdr">Prediction</div>',
                        unsafe_allow_html=True)
 
            sev_col = SEV_COLORS.get(r["severity"], "#aaaaaa")
            if r["severity"] == "Critical":
                st.error(f"CRITICAL — Full road closure expected. Confidence: {r['conf']:.0%}")
            elif r["severity"] == "High" or r["alert_high"]:
                st.error(f"HIGH severity. High/Critical prob: {r['prob_high']:.0%} (threshold 0.30)")
            elif r["severity"] == "Moderate":
                st.warning(f"MODERATE — Heavy delays likely. Confidence: {r['conf']:.0%}")
            else:
                st.success(f"LOW severity. Traffic will flow with minor delays.")
 
            # Severity probability bar
            slabels_ordered = ["Low","Moderate","High","Critical"]
            sev_labels_map  = models.get("sev_labels",{0:"Low",1:"Moderate",2:"High",3:"Critical"})
            sev_order_idx   = {v:k for k,v in sev_labels_map.items()}
 
            fig_b, ax_b = plt.subplots(figsize=(4, 1.4))
            bar_colors = [SEV_COLORS[l] for l in slabels_ordered
                          if l in sev_order_idx]
            probs_ordered = [float(r["sev_prob"][sev_order_idx[l]])
                             for l in slabels_ordered if l in sev_order_idx]
            ax_b.barh(slabels_ordered[:len(probs_ordered)], probs_ordered,
                      color=bar_colors, height=0.5)
            ax_b.set_xlim(0,1)
            ax_b.set_xlabel("Probability")
            ax_b.axvline(0.30, color="gray", linestyle=":", linewidth=1,
                         label="0.30 threshold")
            ax_b.legend(fontsize=7)
            ax_b.grid(axis="x", alpha=0.3)
            fig_b.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)
 
            st.markdown('<div class="section-hdr">Dispatch Orders</div>',
                        unsafe_allow_html=True)
            dispatch_df = pd.DataFrame({
                "Resource":  ["Officers","Barricades","Tow trucks",
                              "Ambulance on standby"],
                "Count":     [r["personnel"], r["barricades"], r["tow"],
                              1 if "accident" in cause else 0],
                "Action":    ["Deploy now" if r["severity"] in ("High","Critical") else "Standby",
                              "Deploy now","Deploy now" if r["tow"]>0 else "Not needed",
                              "Deploy now" if "accident" in cause else "Not needed"],
            })
            st.dataframe(dispatch_df, hide_index=True, use_container_width=True)
 
            st.markdown('<div class="section-hdr">Diversion Plan</div>',
                        unsafe_allow_html=True)
            st.caption(f"Corridor-specific steps for {corridor}")
            for i, step in enumerate(r["div_steps"], 1):
                st.markdown(f"**Step {i}.** {step}")
 
            st.markdown('<div class="section-hdr">Route Summary</div>',
                        unsafe_allow_html=True)
            if r["path"]:
                st.markdown(f"**Route:** {' → '.join(r['path'])}")
                st.markdown(f"**ETA:** {r['eta']:.0f} min  (+{r['delay_add']} min vs normal)")
                if r["blocked"]:
                    st.caption(f"Rerouted around {r['blocked']} (removed from graph)")
            else:
                st.error("No viable route found between selected corridors.")
 
            st.markdown('<div class="section-hdr">Affected radius</div>',
                        unsafe_allow_html=True)
            st.markdown(f"Estimated congestion footprint: **{r['radius']:.2f} km**")
 
        # ── Map panel ───────────────────────────────────────────────────
        with right:
            st.markdown('<div class="section-hdr">Dynamic Rerouting Map</div>',
                        unsafe_allow_html=True)
            st.caption("Node colour: green = low congestion probability, "
                       "orange = moderate, red = high. Active route shown in blue.")
 
            m = folium.Map(location=[12.97,77.59], zoom_start=12,
                           tiles="CartoDB positron")
 
            # All edges faint
            for u, v, _ in BASE_EDGES:
                folium.PolyLine([NODE_COORDS[u],NODE_COORDS[v]],
                                color="#cccccc",weight=1.5,opacity=0.35).add_to(m)
 
            # Nodes coloured by congestion probability
            for node, coord in NODE_COORDS.items():
                p = r["prob_dict"].get(node, 0)
                nc = "#ef476f" if p>0.6 else "#ffd166" if p>0.3 else "#2dc653"
                folium.CircleMarker(
                    coord, radius=6, color=nc,
                    fill=True, fill_opacity=0.8,
                    tooltip=f"{node} — congestion prob {p:.0%}",
                ).add_to(m)
 
            # Blocked marker
            if r["blocked"] and r["blocked"] in NODE_COORDS:
                folium.Marker(
                    NODE_COORDS[r["blocked"]],
                    tooltip=f"Blocked: {r['blocked']}",
                    icon=folium.Icon(color="red", icon="ban", prefix="fa"),
                ).add_to(m)
 
            # Route
            if r["path"]:
                coords_path = [NODE_COORDS[n] for n in r["path"]]
                folium.PolyLine(coords_path, color="#3a86ff",
                                weight=6, opacity=0.9).add_to(m)
                folium.Marker(
                    coords_path[0],
                    tooltip=f"Start: {r['path'][0]}",
                    icon=folium.Icon(color="green",icon="play",prefix="fa"),
                ).add_to(m)
                folium.Marker(
                    coords_path[-1],
                    tooltip=f"End: {r['path'][-1]}",
                    icon=folium.Icon(color="blue",icon="flag",prefix="fa"),
                ).add_to(m)
 
            # Hotspot markers
            for name, (ht_lat, ht_lon) in HOTSPOTS.items():
                folium.CircleMarker(
                    [ht_lat,ht_lon], radius=8,
                    color="#000000", fill=True, fill_opacity=0.4,
                    tooltip=f"Hotspot: {name.replace('_',' ').title()}",
                ).add_to(m)
 
            st_folium(m, width="100%", height=520)
 
        # ── Feedback ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Post-event confirmation")
        st.caption("After the event resolves, confirm the actual outcome. "
                   "This closes the learning loop and improves future predictions.")
 
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            actual_sev = st.selectbox("Actual severity observed",
                                      ["Low","Moderate","High","Critical"],
                                      index=["Low","Moderate","High","Critical"]
                                      .index(r["severity"]))
        with fc2:
            actual_dur = st.slider("Actual duration (hrs)", 0.5, 8.0, dur, 0.5,
                                   key="fb_actual_dur")
        with fc3:
            st.markdown("<br>", unsafe_allow_html=True)
            fb_btn = st.button("Submit feedback", use_container_width=True,
                               disabled=st.session_state.fb_done)
            if fb_btn:
                record = {
                    "timestamp":        datetime.utcnow().isoformat(),
                    "corridor":         corridor,
                    "event_cause":      cause,
                    "hour":             hour,
                    "day_of_week":      day,
                    "priority":         priority,
                    "predicted_sev":    r["severity"],
                    "actual_sev":       actual_sev,
                    "prob_high":        r["prob_high"],
                    "personnel_pred":   r["personnel"],
                    "barricades_pred":  r["barricades"],
                    "actual_duration":  actual_dur,
                    "correct":          r["severity"] == actual_sev,
                }
                append_log(record)
                st.session_state.fb_done = True
                st.success("Feedback recorded. View trends in the Feedback and Learning tab.")
            if st.session_state.fb_done:
                st.info("Feedback already submitted for this run.")
 
# TAB 2 — IMPACT DASHBOARD
with tab_impact:
    st.markdown("## Corridor Incident Analysis")
    st.caption("All statistics computed directly from the ASTraM operational dataset.")
 
    if df_raw is None:
        st.warning(
            f"Dataset CSV not found at `{CSV_PATH}`. "
            "Place the CSV in the same folder as app.py and restart."
        )
    else:
        # ── Compute corridor stats from real data ──────────────────────
        corr_df = df_raw[df_raw["corridor"] != "Non-corridor"].copy()
 
        stats = (
            corr_df.groupby("corridor")
            .agg(
                total_incidents   = ("corridor",              "count"),
                closure_rate      = ("requires_road_closure", "mean"),
                high_priority_pct = ("priority",              lambda x: (x == "High").mean()),
                top_cause         = ("event_cause",           lambda x: x.value_counts().idxmax()),
            )
            .reset_index()
            .sort_values("total_incidents", ascending=False)
        )
        stats["closure_rate_pct"]      = (stats["closure_rate"]      * 100).round(1)
        stats["high_priority_pct_disp"]= (stats["high_priority_pct"] * 100).round(1)
 
        # ── KPI row ────────────────────────────────────────────────────
        total_incidents  = int(corr_df.shape[0])
        worst_corridor   = stats.iloc[0]["corridor"]
        worst_count      = int(stats.iloc[0]["total_incidents"])
        highest_closure  = stats.sort_values("closure_rate", ascending=False).iloc[0]
 
        k1, k2, k3 = st.columns(3)
        k1.metric("Total corridor incidents", f"{total_incidents:,}")
        k2.metric("Most incident-prone corridor", worst_corridor,
                  delta=f"{worst_count} incidents")
        k3.metric("Highest closure rate",
                  highest_closure["corridor"],
                  delta=f"{highest_closure['closure_rate_pct']}% of incidents closed road")
 
        st.markdown("---")
 
        # ── Plot 1: Incident count per corridor ────────────────────────
        top_stats = stats.head(15)
        colors_i  = ["#ef476f" if v > stats["total_incidents"].quantile(0.75)
                     else "#ffd166" if v > stats["total_incidents"].median()
                     else "#2dc653"
                     for v in top_stats["total_incidents"]]
 
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.barh(top_stats["corridor"][::-1],
                 top_stats["total_incidents"][::-1],
                 color=colors_i[::-1])
        ax1.axvline(stats["total_incidents"].mean(), color="gray",
                    linestyle="--", alpha=0.6, label="Average")
        ax1.set_title("Incident count per corridor (top 15)", fontweight="bold")
        ax1.set_xlabel("Number of incidents")
        ax1.legend(fontsize=9)
        ax1.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
 
        st.markdown("---")
 
        # ── Plot 2: Closure rate + high-priority rate side by side ─────
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Corridor risk profile — from real incident data",
                      fontsize=13, fontweight="bold")
 
        top_closure = stats.sort_values("closure_rate", ascending=False).head(12)
        axes2[0].barh(top_closure["corridor"][::-1],
                      top_closure["closure_rate_pct"][::-1],
                      color="#ef476f")
        axes2[0].set_title("Road closure rate (%) per corridor")
        axes2[0].set_xlabel("% of incidents that required road closure")
        axes2[0].grid(axis="x", alpha=0.3)
 
        top_hp = stats.sort_values("high_priority_pct", ascending=False).head(12)
        axes2[1].barh(top_hp["corridor"][::-1],
                      top_hp["high_priority_pct_disp"][::-1],
                      color="#ffd166")
        axes2[1].set_title("High-priority incident rate (%) per corridor")
        axes2[1].set_xlabel("% of incidents flagged as High priority")
        axes2[1].grid(axis="x", alpha=0.3)
 
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
 
        st.markdown("---")
 
        # ── Plot 3: Top event cause per corridor (heatmap-style) ───────
        st.markdown("### Most common event cause per corridor")
        cause_pivot = (
            corr_df.groupby(["corridor", "event_cause"])
            .size()
            .reset_index(name="count")
        )
        top_corridors = stats.head(12)["corridor"].tolist()
        cause_pivot   = cause_pivot[cause_pivot["corridor"].isin(top_corridors)]
        cause_matrix  = (
            cause_pivot.pivot_table(index="corridor", columns="event_cause",
                                    values="count", fill_value=0)
        )
        # Keep only top 6 causes for readability
        top_causes    = cause_pivot.groupby("event_cause")["count"].sum()\
                                   .nlargest(6).index.tolist()
        cause_matrix  = cause_matrix[[c for c in top_causes if c in cause_matrix.columns]]
 
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        im = ax3.imshow(cause_matrix.values, aspect="auto", cmap="YlOrRd")
        ax3.set_xticks(range(len(cause_matrix.columns)))
        ax3.set_xticklabels(cause_matrix.columns, rotation=30, ha="right", fontsize=10)
        ax3.set_yticks(range(len(cause_matrix.index)))
        ax3.set_yticklabels(cause_matrix.index, fontsize=9)
        plt.colorbar(im, ax=ax3, label="Incident count")
        ax3.set_title("Incident heatmap — corridor vs event cause (top 12 corridors, top 6 causes)",
                      fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
 
        # ── Full stats table ───────────────────────────────────────────
        st.markdown("### Full corridor statistics table")
        display_cols = {
            "corridor":             "Corridor",
            "total_incidents":      "Total incidents",
            "closure_rate_pct":     "Closure rate (%)",
            "high_priority_pct_disp": "High priority (%)",
            "top_cause":            "Most common cause",
        }
        st.dataframe(
            stats[list(display_cols.keys())].rename(columns=display_cols),
            hide_index=True,
            use_container_width=True,
        )
 
# TAB 3 — SHAP EXPLAINABILITY
with tab_shap:
    st.markdown("## AI Explainability")
 
    if not st.session_state.sim_run or st.session_state.sim_result is None:
        st.info("Run a simulation first to see the prediction breakdown.")
    elif not SHAP_OK:
        st.warning("Install the shap package to enable this tab: pip install shap")
    elif "shap_explainer" not in models:
        st.warning("shap_explainer.pkl not found in models/. Re-run the notebook.")
    else:
        r   = st.session_state.sim_result
        exp = models["shap_explainer"]
        inp = r["inp_df"]
 
        try:
            sv = exp.shap_values(inp)
 
            sev_labels_map = models.get("sev_labels", {0:"Low",1:"Moderate",2:"High",3:"Critical"})
            sev_to_idx     = {v: k for k, v in sev_labels_map.items()}
            cls_idx        = int(sev_to_idx.get(r["severity"], 2))
 
            # SHAP returns different shapes depending on version and model type:
            #   - list of arrays: sv[class_idx] has shape (n_samples, n_features)
            #   - single ndarray shape (n_samples, n_features, n_classes)  [newer SHAP]
            #   - single ndarray shape (n_classes, n_samples, n_features)  [some versions]
            sv_arr = np.array(sv)
 
            if isinstance(sv, list):
                # list[n_classes] each (n_samples, n_features)
                sv_row = np.array(sv[cls_idx])[0]
            elif sv_arr.ndim == 3:
                if sv_arr.shape[0] == len(sev_labels_map):
                    # (n_classes, n_samples, n_features)
                    sv_row = sv_arr[cls_idx, 0, :]
                else:
                    # (n_samples, n_features, n_classes)
                    sv_row = sv_arr[0, :, cls_idx]
            elif sv_arr.ndim == 2:
                # (n_samples, n_features) — single output or already squeezed
                sv_row = sv_arr[0]
            else:
                sv_row = sv_arr.flatten()
 
            sv_row = np.array(sv_row).flatten()
            feat_shap = pd.Series(sv_row, index=inp.columns).sort_values()
 
            # Top 8 positive and negative
            top_pos = feat_shap.nlargest(8)
            top_neg = feat_shap.nsmallest(8)
            combined = pd.concat([top_neg, top_pos]).sort_values()
 
            fig_s, ax_s = plt.subplots(figsize=(8, 5))
            colors_s = ["#ef476f" if v > 0 else "#2dc653" for v in combined.values]
            ax_s.barh(combined.index, combined.values, color=colors_s)
            ax_s.axvline(0, color="black", linewidth=0.8)
            ax_s.set_title(
                f"SHAP values — {r['severity']} prediction for this event",
                fontsize=11, fontweight="bold"
            )
            ax_s.set_xlabel("SHAP value (positive = pushes toward higher severity)")
            ax_s.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)
 
            st.markdown("### Plain-language breakdown")
            st.markdown(
                f"The model predicted **{r['severity']}** with "
                f"**{r['conf']:.0%}** confidence."
            )
 
            st.markdown("**Features increasing severity:**")
            for feat, val in top_pos.items():
                clean = feat.replace("_"," ").replace(" enc","")
                st.markdown(f"- {clean} (+{val:.3f})")
 
            st.markdown("**Features reducing severity:**")
            for feat, val in top_neg.items():
                clean = feat.replace("_"," ").replace(" enc","")
                st.markdown(f"- {clean} ({val:.3f})")
 
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            st.caption("This can happen if the explainer was saved with a "
                       "different model version. Re-run the notebook to regenerate.")
 
# TAB 4 — FEEDBACK AND LEARNING
with tab_log:
    st.markdown("## Post-event Learning System")
    st.caption("Every confirmed event is logged. "
               "The accuracy trend shows whether the model is drifting on real operational data.")
 
    df_log = load_log()
 
    if df_log.empty:
        st.info("No events logged yet. Submit the first simulation result "
                "via the Post-event confirmation section.")
    else:
        df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])
        df_log = df_log.sort_values("timestamp").reset_index(drop=True)
        df_log["rolling_acc"] = df_log["correct"].expanding().mean()
 
        total   = len(df_log)
        correct = int(df_log["correct"].sum())
        acc     = correct / total
 
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Events logged",       total)
        m2.metric("Correct predictions", correct)
        m3.metric("Overall accuracy",    f"{acc:.1%}")
        m4.metric("Last 5 accuracy",     f"{df_log['correct'].tail(5).mean():.1%}")
 
        st.markdown("---")
 
        fig_log, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig_log.suptitle("Post-Event Learning Dashboard", fontweight="bold")
 
        axes[0].plot(df_log.index, df_log["rolling_acc"]*100,
                     color="#3a86ff", linewidth=2, marker="o", markersize=4)
        axes[0].axhline(85, color="gray", linestyle="--", alpha=0.6, label="Target 85%")
        axes[0].set_title("Cumulative accuracy")
        axes[0].set_xlabel("Event number")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim(0,105)
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)
 
        if "prob_high" in df_log.columns and "actual_sev" in df_log.columns:
            act_high = df_log["actual_sev"].isin(["High","Critical"]).astype(int)
            jit = np.random.uniform(-0.04,0.04,len(df_log))
            sc_colors = df_log["correct"].map({True:"#2dc653",False:"#ef476f"})
            axes[1].scatter(df_log["prob_high"], act_high+jit,
                            c=sc_colors, alpha=0.7, s=55,
                            edgecolors="white", linewidth=0.5)
            axes[1].axvline(0.30, color="gray", linestyle=":", label="0.30 threshold")
            axes[1].set_title("Calibration: prob_high vs actual")
            axes[1].set_xlabel("Predicted High/Critical probability")
            axes[1].set_ylabel("Actually High/Critical")
            axes[1].set_yticks([0,1])
            axes[1].set_yticklabels(["No","Yes"])
            axes[1].legend(fontsize=9)
            axes[1].grid(alpha=0.3)
 
        if "corridor" in df_log.columns and df_log["corridor"].nunique() > 1:
            ca = df_log.groupby("corridor")["correct"].mean().sort_values()
            c_colors = ["#ef476f" if v<0.6 else "#ffd166" if v<0.8
                        else "#2dc653" for v in ca.values]
            axes[2].barh(ca.index, ca.values*100, color=c_colors)
            axes[2].axvline(85, color="gray", linestyle="--", alpha=0.6)
            axes[2].set_title("Accuracy by corridor")
            axes[2].set_xlabel("Accuracy (%)")
            axes[2].set_xlim(0,105)
            axes[2].grid(axis="x", alpha=0.3)
        else:
            axes[2].text(0.5,0.5,"More data needed",
                         ha="center",va="center",transform=axes[2].transAxes,
                         color="gray")
            axes[2].set_title("Accuracy by corridor")
 
        plt.tight_layout()
        st.pyplot(fig_log, use_container_width=True)
        plt.close(fig_log)
 
        st.markdown("---")
        st.markdown("### Event log (last 50)")
        cols_show = [c for c in ["timestamp","corridor","event_cause","priority",
                                  "predicted_sev","actual_sev","prob_high","correct"]
                     if c in df_log.columns]
        st.dataframe(df_log[cols_show].sort_values("timestamp",ascending=False)
                     .head(50), hide_index=True, use_container_width=True)
 
        dl, cl = st.columns(2)
        with dl:
            st.download_button("Download log as CSV",
                               df_log.to_csv(index=False).encode(),
                               "astram_log.csv","text/csv",
                               use_container_width=True)
        with cl:
            if st.button("Clear log", use_container_width=True):
                if os.path.exists(LOG_FILE):
                    os.remove(LOG_FILE)
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px;'>"
    "ASTraM · Gridlock Hackathon 2.0 · XGBoost · SHAP · NetworkX · Folium"
    "</div>",
    unsafe_allow_html=True,
)