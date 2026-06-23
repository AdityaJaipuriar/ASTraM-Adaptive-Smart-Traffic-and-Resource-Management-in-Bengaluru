"""
ASTraM — Adaptive Smart Traffic & Resource Management
Streamlit deployment app.
"""
 
import os
import json
from datetime import datetime
 
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
 
# CONFIG
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))
 
st.set_page_config(
    page_title="ASTraM — Traffic Event Console",
    page_icon="🚦",
    layout="wide",
)
 
def artifact_path(name: str) -> str:
    return os.path.join(ARTIFACT_DIR, name)

# ARTEFACT LOADING
@st.cache_resource
def load_artifacts():
    missing = []
    required = [
        'xgboost_traffic_brain.pkl', 'model_features.pkl',
        'resource_model_manpower.pkl', 'resource_model_barricades.pkl',
        'resource_model_tow.pkl', 'resource_features.pkl',
        'corridor_stats.pkl', 'hotspots.pkl',
        'diversion_templates.pkl', 'deploy_threshold.pkl',
    ]
    for f in required:
        if not os.path.exists(artifact_path(f)):
            missing.append(f)
    if missing:
        return None, missing
 
    artifacts = {
        'closure_model':       joblib.load(artifact_path('xgboost_traffic_brain.pkl')),
        'model_features':      joblib.load(artifact_path('model_features.pkl')),
        'manpower_model':      joblib.load(artifact_path('resource_model_manpower.pkl')),
        'barricades_model':    joblib.load(artifact_path('resource_model_barricades.pkl')),
        'tow_model':           joblib.load(artifact_path('resource_model_tow.pkl')),
        'resource_features':   joblib.load(artifact_path('resource_features.pkl')),
        'corridor_stats':      joblib.load(artifact_path('corridor_stats.pkl')),
        'hotspots':            joblib.load(artifact_path('hotspots.pkl')),
        'diversion_templates': joblib.load(artifact_path('diversion_templates.pkl')),
        'deploy_threshold':    joblib.load(artifact_path('deploy_threshold.pkl')),
    }
    return artifacts, []
 
ART, MISSING = load_artifacts()
 
if ART is None:
    st.title("ASTraM — Traffic Event Console")
    st.error(
        "Some model artefacts are missing. Ensure all required .pkl files are in the main directory."
    )
    for f in MISSING:
        st.write(f"- `{f}`")
    st.stop()

# CORE LOGIC
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))
 
def build_feature_row(event: dict, feature_cols: list, corridor_rates: dict,
                       zone_rates: dict, hotspots: dict, closure_value=None) -> pd.Series:
    row = pd.Series(0, index=feature_cols, dtype=float)
    row['latitude']       = event['latitude']
    row['longitude']      = event['longitude']
    row['priority']       = 1 if event.get('priority', 'Low') == 'High' else 0
    row['duration_hrs']   = event.get('duration_hrs', 2.0)
    row['Hour_of_Day']    = event['hour']
    row['Day_of_Week']    = event['day_of_week']
    row['Month']          = event.get('month', datetime.now().month)
    row['Is_Weekend']     = event.get('is_weekend', 0)
    row['Is_Rush_Hour']   = 1 if (7 <= event['hour'] <= 10 or 17 <= event['hour'] <= 20) else 0
    row['geo_cluster']    = event.get('geo_cluster', 0)

    # Data-Driven Closure Rates
    if 'corridor_closure_rate' in row.index and corridor_rates:
        default_corr = float(np.median(list(corridor_rates.values())))
        row['corridor_closure_rate'] = corridor_rates.get(event['corridor'], default_corr)

    if 'zone_closure_rate' in row.index and zone_rates:
        default_zone = float(np.median(list(zone_rates.values())))
        row['zone_closure_rate'] = zone_rates.get(event.get('zone', 'Zone_Unknown'), default_zone)

    if closure_value is not None and 'requires_road_closure' in row.index:
        row['requires_road_closure'] = closure_value

    for name, (hlat, hlon) in hotspots.items():
        col = f'dist_{name}_km'
        if col in row.index:
            row[col] = haversine_km(event['latitude'], event['longitude'], hlat, hlon)

    for prefix, value in [
        ('event_cause_',    event['event_cause']),
        ('corridor_',       event['corridor']),
        ('veh_type_',       event.get('veh_type')),
        ('event_type_',     event.get('event_type', 'unplanned')),
        ('zone_',           event.get('zone', 'Zone_Unknown')),
        ('corridor_zone_',  f"{event['corridor']} | {event.get('zone', 'Zone_Unknown')}"),
    ]:
        col = f'{prefix}{value}'
        if col in row.index:
            row[col] = 1

    return row
 
def get_diversion(corridor_name: str, templates: dict):
    name = str(corridor_name).strip()
    if name == 'Non-corridor' or not name:
        return templates['default']
    for key in templates:
        if key == 'default':
            continue
        if key.lower() in name.lower():
            return templates[key]
    return templates['default']
 
def get_corridor_history(corridor: str) -> dict:
    """Return real historical stats for the corridor from the saved dataset summary."""
    cs = ART['corridor_stats']
    row = cs[cs['corridor'] == corridor]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {
        'total_incidents':    int(r['total_incidents']),
        'closure_pct':        float(r['closure_pct']),
        'high_priority_pct':  float(r['high_priority_pct']),
        'peak_hour_pct':      float(r['peak_hour_pct']),
        'most_common_cause':  str(r['most_common_cause']),
    }
 
def recommend(event: dict) -> dict:
    # Calculate rates dynamically from the existing corridor_stats file
    cs_df = ART['corridor_stats']
    # Convert the closure percentage (e.g. 45.5%) into a 0-1 rate (e.g. 0.455)
    dynamic_corridor_rates = dict(zip(cs_df['corridor'], cs_df['closure_pct'] / 100.0))
    dynamic_zone_rates = {} # We leave this empty, the row builder will safely handle it!

    closure_row = build_feature_row(event, ART['model_features'], 
                                    dynamic_corridor_rates, dynamic_zone_rates, ART['hotspots'])
    
    closure_row_df = pd.DataFrame([closure_row])[ART['model_features']]
    closure_prob   = float(ART['closure_model'].predict_proba(closure_row_df)[0, 1])
    closure_pred   = int(closure_prob >= ART['deploy_threshold'])

    res_row    = build_feature_row(event, ART['resource_features'],
                                   dynamic_corridor_rates, dynamic_zone_rates, ART['hotspots'],
                                   closure_value=closure_pred)
                                   
    res_row_df = pd.DataFrame([res_row])[ART['resource_features']]
    manpower   = int(round(ART['manpower_model'].predict(res_row_df)[0]))
    barricades = int(round(ART['barricades_model'].predict(res_row_df)[0]))
    tow        = int(round(ART['tow_model'].predict(res_row_df)[0]))

    diverts = 1 if closure_prob < 0.3 else (2 if closure_prob < 0.6 else 3)
    plan    = get_diversion(event.get('corridor', ''), ART['diversion_templates'])[:diverts + 1]

    return {
        'closure_probability': closure_prob,
        'closure_predicted':   bool(closure_pred),
        'manpower_required':   manpower,
        'barricade_sections':  barricades,
        'tow_trucks_required': tow,
        'diversion_plan':      plan,
    }

# FEEDBACK LOG
LOG_FILE = artifact_path('event_feedback_log.json')
 
def log_event(record: dict):
    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            log = json.load(f)
    log.append(record)
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)
 
def load_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    with open(LOG_FILE, 'r') as f:
        log = json.load(f)
    return pd.DataFrame(log)

# UI (User Interface)
st.title("ASTraM — Traffic Event Console")
st.caption(
    "Predict whether a logged traffic event will require a road closure, "
    "how many resources to deploy, what the historical risk profile of the "
    "corridor looks like, and what to tell the units on the ground."
)
 
tab_predict, tab_corridor, tab_map, tab_log = st.tabs([
    "New Event", "Corridor Risk Profile", "Corridor Map", "Model Accuracy Log"
])
 
# Safely extract cause options from model features
CAUSE_OPTIONS = sorted([c.replace('event_cause_', '') for c in ART['model_features'] if c.startswith('event_cause_')])
if not CAUSE_OPTIONS:
    CAUSE_OPTIONS = ['accident', 'vehicle_breakdown', 'vip_movement', 'public_event', 'water_logging']

CORRIDOR_OPTIONS = sorted(ART['corridor_stats']['corridor'].tolist()) + ['Non-corridor']
ZONE_OPTIONS     = ['Zone_Unknown', 'Central Zone 1', 'Central Zone 2',
                    'East Zone 1', 'East Zone 2', 'North Zone 1', 'North Zone 2',
                    'South Zone 1', 'South Zone 2', 'West Zone 1', 'West Zone 2']
VEH_TYPE_OPTIONS = ['No_Vehicle_or_Unknown', 'bmtc_bus', 'ksrtc_bus', 'private_bus',
                    'heavy_vehicle', 'lcv', 'truck', 'private_car', 'taxi', 'auto', 'others']

# TAB 1 — NEW EVENT
with tab_predict:
    left, right = st.columns([1, 1.3])
 
    with left:
        st.subheader("Event details")
        event_cause  = st.selectbox("Event cause", CAUSE_OPTIONS,
                                    index=CAUSE_OPTIONS.index('vehicle_breakdown')
                                    if 'vehicle_breakdown' in CAUSE_OPTIONS else 0)
        corridor     = st.selectbox("Corridor", CORRIDOR_OPTIONS)
        zone         = st.selectbox("Zone", ZONE_OPTIONS)
        veh_type     = st.selectbox("Vehicle type", VEH_TYPE_OPTIONS)
        priority     = st.radio("Priority", ['Low', 'High'], horizontal=True)
        event_type   = st.radio("Event type", ['unplanned', 'planned'], horizontal=True)
 
        col_a, col_b = st.columns(2)
        with col_a:
            hour        = st.slider("Hour of day", 0, 23, 18)
            day_of_week = st.selectbox(
                "Day of week",
                options=list(range(7)),
                format_func=lambda d: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d],
                index=2,
            )
        with col_b:
            duration_hrs = st.number_input("Expected duration (hrs)", 0.1, 48.0, 2.0, 0.5)
            is_weekend   = 1 if day_of_week >= 5 else 0
 
        st.subheader("Location")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude  = st.number_input("Latitude",  12.80, 13.30, 12.976, format="%.4f")
        with col_lon:
            longitude = st.number_input("Longitude", 77.30, 77.80, 77.660, format="%.4f")
 
        run = st.button("Get recommendation", type="primary", use_container_width=True)
 
    with right:
        st.subheader("Output")
        if run:
            event = {
                'event_cause': event_cause, 'corridor': corridor, 'zone': zone,
                'veh_type': veh_type, 'event_type': event_type, 'priority': priority,
                'hour': hour, 'day_of_week': day_of_week,
                'month': datetime.now().month, 'is_weekend': is_weekend,
                'latitude': latitude, 'longitude': longitude,
                'geo_cluster': 0, 'duration_hrs': duration_hrs,
            }
            result = recommend(event)
            hist   = get_corridor_history(corridor)
 
            # Closure prediction 
            badge = "ROAD CLOSURE LIKELY" if result['closure_predicted'] else "closure unlikely"
            st.metric("Closure probability",
                      f"{result['closure_probability']*100:.1f}%", badge)
 
            # Resource dispatch
            m1, m2, m3 = st.columns(3)
            m1.metric("Manpower",          result['manpower_required'])
            m2.metric("Barricade sections",result['barricade_sections'])
            m3.metric("Tow trucks",        result['tow_trucks_required'])
 
            # Historical corridor context from real data
            st.markdown("#### Historical risk profile for this corridor")
            if hist:
                h1, h2, h3 = st.columns(3)
                h1.metric("Past incidents recorded",   hist['total_incidents'])
                h2.metric("Historical closure rate",   f"{hist['closure_pct']}%")
                h3.metric("High-priority rate",        f"{hist['high_priority_pct']}%")
 
                h4, h5 = st.columns(2)
                h4.metric("Most common cause",         hist['most_common_cause'])
                h5.metric("Incidents during peak hours", f"{hist['peak_hour_pct']}%")
 
                # Mini bar chart — closure vs non-closure split from real data
                fig_h, ax_h = plt.subplots(figsize=(4, 1.2))
                closed_pct  = hist['closure_pct']
                open_pct    = 100 - closed_pct
                ax_h.barh([0], [open_pct],   color='#2dc653', height=0.5, label='No closure')
                ax_h.barh([0], [closed_pct], color='#ef476f', height=0.5,
                          left=[open_pct], label='Closure')
                ax_h.set_xlim(0, 100)
                ax_h.set_yticks([])
                ax_h.set_xlabel("% of past incidents")
                ax_h.legend(fontsize=8, loc='upper right')
                ax_h.set_title("Closure split — historical", fontsize=9)
                fig_h.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig_h, use_container_width=True)
                plt.close(fig_h)
 
                st.caption(
                    f"Source: ASTraM operational dataset — {hist['total_incidents']} "
                    f"real events recorded on {corridor}."
                )
            else:
                st.info("No historical records found for this corridor in the dataset.")
 
            # Diversion plan 
            st.markdown("#### Diversion plan")
            for i, step in enumerate(result['diversion_plan'], 1):
                st.write(f"{i}. {step}")
 
            # Feedback log
            st.divider()
            st.markdown("##### Log outcome (after the event resolves)")
            actual_closure = st.radio(
                "Did this event actually require a road closure?",
                ['Not yet known', 'Yes', 'No'],
                horizontal=True, key='actual_closure_radio'
            )
            if actual_closure != 'Not yet known':
                if st.button("Save to feedback log"):
                    record = {
                        'timestamp':              datetime.utcnow().isoformat(),
                        'corridor':               corridor,
                        'event_cause':            event_cause,
                        'hour':                   hour,
                        'day_of_week':            day_of_week,
                        'priority':               priority,
                        'predicted_closure':      int(result['closure_predicted']),
                        'actual_closure':         1 if actual_closure == 'Yes' else 0,
                        'prediction_probability': result['closure_probability'],
                        'manpower_pred':          result['manpower_required'],
                        'barricades_pred':        result['barricade_sections'],
                        'tow_trucks_pred':        result['tow_trucks_required'],
                        'correct':                int(result['closure_predicted']) ==
                                                  (1 if actual_closure == 'Yes' else 0),
                    }
                    log_event(record)
                    st.success("Logged. See the Model Accuracy Log tab.")
        else:
            st.info("Fill in the event details and click 'Get recommendation'.")
 
# TAB 2 — CORRIDOR RISK PROFILE (real data)
with tab_corridor:
    st.subheader("Corridor Risk Profile")
    st.caption("All figures computed directly from the ASTraM operational dataset.")
 
    cs = ART['corridor_stats'].sort_values('total_incidents', ascending=False)
 
    k1, k2, k3 = st.columns(3)
    k1.metric("Total corridors tracked",    len(cs))
    k2.metric("Most incident-prone",        cs.iloc[0]['corridor'],
              delta=f"{int(cs.iloc[0]['total_incidents'])} incidents")
    worst_closure = cs.sort_values('closure_rate' if 'closure_rate' in cs.columns else 'closure_pct', ascending=False).iloc[0]
    k3.metric("Highest closure rate",       worst_closure['corridor'],
              delta=f"{worst_closure['closure_pct']}% of incidents")
 
    st.markdown("---")
 
    fig_c, axes_c = plt.subplots(1, 3, figsize=(17, 6))
    fig_c.suptitle("Corridor Risk Profile — from real incident records",
                   fontsize=13, fontweight="bold")
 
    top15 = cs.head(15)
 
    ax = axes_c[0]
    ax.barh(top15['corridor'][::-1], top15['total_incidents'][::-1], color='#3a86ff')
    ax.axvline(cs['total_incidents'].mean(), color='gray',
               linestyle='--', alpha=0.6, label='Average')
    ax.set_title("Incident count (top 15)")
    ax.set_xlabel("Number of incidents")
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)
 
    ax = axes_c[1]
    top_cl = cs.sort_values('closure_rate' if 'closure_rate' in cs.columns else 'closure_pct', ascending=False).head(12)
    colors1 = ['#ef476f' if v > 60 else '#ffd166' if v > 30 else '#2dc653'
               for v in top_cl['closure_pct']]
    ax.barh(top_cl['corridor'][::-1], top_cl['closure_pct'][::-1], color=colors1[::-1])
    ax.set_title("Road closure rate %")
    ax.set_xlabel("% of incidents requiring closure")
    ax.grid(axis='x', alpha=0.3)
 
    ax = axes_c[2]
    top_hp = cs.sort_values('high_priority_rate' if 'high_priority_rate' in cs.columns else 'high_priority_pct', ascending=False).head(12)
    ax.barh(top_hp['corridor'][::-1], top_hp['high_priority_pct'][::-1], color='#8338ec')
    ax.set_title("High-priority incident rate %")
    ax.set_xlabel("% flagged as High priority")
    ax.grid(axis='x', alpha=0.3)
 
    plt.tight_layout()
    st.pyplot(fig_c, use_container_width=True)
    plt.close(fig_c)
 
    # Cause heatmap 
    st.markdown("### Most common event cause per corridor")
    st.dataframe(
        cs[['corridor', 'total_incidents', 'closure_pct',
            'high_priority_pct', 'peak_hour_pct', 'most_common_cause']]
        .rename(columns={
            'corridor':           'Corridor',
            'total_incidents':    'Total incidents',
            'closure_pct':        'Closure rate (%)',
            'high_priority_pct':  'High priority (%)',
            'peak_hour_pct':      'Peak hour (%)',
            'most_common_cause':  'Most common cause',
        }),
        hide_index=True,
        use_container_width=True,
    )
 
# TAB 3 — CORRIDOR MAP
with tab_map:
    st.subheader("Historical Event Heatmap")
    st.caption("A sampled map of past events across Bengaluru, colour-coded by data-driven severity.")
 
    map_html_path = artifact_path('bengaluru_events_map.html')
    map_pkl_path  = artifact_path('map_sample.pkl')
 
    # OPTION 1: Load the HTML file directly (Fastest & Most Efficient)
    if os.path.exists(map_html_path):
        with open(map_html_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600, scrolling=True)
 
    # OPTION 2: Generate dynamically via st_folium (Requires exporting the dataframe)
    elif os.path.exists(map_pkl_path):
        sample = joblib.load(map_pkl_path)
        COLOR_MAP = {'Low': 'green', 'Moderate': 'orange', 'High': 'red', 'Critical': 'darkred'}
        m = folium.Map(location=[12.97, 77.59], zoom_start=12, tiles='CartoDB positron')
 
        for _, row in sample.iterrows():
            sev   = str(row.get('severity_label', 'Unknown'))
            cause = str(row.get('event_cause', ''))
            corr  = str(row.get('corridor', ''))
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=COLOR_MAP.get(sev, 'gray'),
                fill=True, fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{str(row.get('event_type', '')).title()}</b><br>"
                    f"Cause: {cause}<br>"
                    f"Corridor: {corr}<br>"
                    f"Severity: <b>{sev}</b><br>"
                    f"Closure required: {bool(row.get('requires_road_closure', False))}<br>"
                    f"Duration: {row.get('duration_hrs', 0):.1f} hrs",
                    max_width=220
                )
            ).add_to(m)
 
        legend_html = '''
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
             background:white;padding:12px;border-radius:8px;
             border:1px solid #ccc;font-size:13px;">
          <b>Event severity</b><br>
          <span style="color:green">&#9679;</span> Low<br>
          <span style="color:orange">&#9679;</span> Moderate<br>
          <span style="color:red">&#9679;</span> High<br>
          <span style="color:darkred">&#9679;</span> Critical
        </div>'''
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, width=900, height=550)
 
    else:
        st.warning("Map not found! Please place `bengaluru_events_map.html` in your GitHub repository.")
 
# TAB 4 — MODEL ACCURACY LOG
with tab_log:
    st.subheader("Prediction accuracy over time")
    df_log = load_log()
    if df_log.empty:
        st.info("No logged outcomes yet. Use the 'Log outcome' control on the "
                "New Event tab after an event resolves.")
    else:
        df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
        df_log = df_log.sort_values('timestamp')
        overall_acc = df_log['correct'].mean()
        st.metric("Overall accuracy", f"{overall_acc*100:.1f}%",
                  f"{len(df_log)} logged events")
        df_log['cumulative_accuracy'] = df_log['correct'].expanding().mean()
        st.line_chart(df_log.set_index('timestamp')['cumulative_accuracy'])
        st.dataframe(
            df_log[['timestamp', 'corridor', 'event_cause',
                    'predicted_closure', 'actual_closure', 'correct']]
            .sort_values('timestamp', ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "Download log as CSV",
            df_log.to_csv(index=False).encode(),
            "astram_feedback_log.csv", "text/csv",
            use_container_width=True,
        )
 
st.divider()
st.caption(
    "ASTraM — model trained on the anonymised Bengaluru event export. "
    "Predictions are decision support, not a substitute for on-ground judgement."
)