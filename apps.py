import pandas as pd
import ieee738
import math
from ieee738.ieee738 import ConductorParams, Conductor
import numpy as np
import matplotlib.pyplot as plt
import json
import folium
import streamlit as st
from streamlit_folium import st_folium
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import re
import google.generativeai as genai


st.set_page_config(
    page_title="Oʻahu Line Ratings",
    page_icon="", # I put the icon back
    layout="wide",
)
st.title("Dynamic Line Ratings Dashboard")
st.caption("Explore how temperature and wind affect transmission line performance on Oʻahu.")

# --- Sidebar Inputs ---
st.sidebar.header("Scenario Inputs")
Ta = st.sidebar.slider("Temperature (°C)", 20, 60, 35)
wind = st.sidebar.slider("Wind speed (ft/s)", 0.0, 15.0, 2.0, 0.5)
wind_dir_from = st.sidebar.slider("Wind direction (° from North)", 0, 359, 270)  # 270 = from West
sun_time = st.sidebar.slider("Sun time (hours)", 0, 12, 12)

# --- API KEY SECTION ---
# Per your request, the sidebar header is removed and the key is left as is.
API_KEY = ""

GEMINI_READY = False
model = None
generation_config = None

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        generation_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=1,
            top_k=1,
            max_output_tokens=2048,
        )
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            generation_config=generation_config,
        )
        GEMINI_READY = True
    except Exception as e:
        st.error(f"Error configuring AI model: {e}")
# --- END OF API KEY SECTION ---

arrow_dir = (wind_dir_from + 180) % 360


# --- CORE FUNCTIONS ---

# This function is fast, so it doesn't need caching itself.
def calc_dynamic_rating(row, Ta_C, wind_ft_s, wind_angle_deg=90, sun_time=12, date_str="12 Jun"):
    """Calculates the steady-state thermal rating for a given conductor row and weather."""
    ambient_defaults = {
        'Ta': Ta_C, 'WindVelocity': wind_ft_s, 'WindAngleDeg': wind_angle_deg,
        'SunTime': sun_time, 'Date': date_str, 'Emissivity': 0.8, 'Absorptivity': 0.8,
        'Direction': 'EastWest', 'Atmosphere': 'Clear', 'Elevation': 1000, 'Latitude': 27,
    }
    acsr = {
        'TLo': 25, 'THi': 50, 'RLo': row.RLo_ohm_per_ft, 'RHi': row.RHi_ohm_per_ft,
        'Diameter': row.Diameter_in, 'Tc': row.MOT_C,
    }
    cp = ConductorParams(**ambient_defaults, **acsr)
    con = Conductor(cp)
    rating_amps = con.steady_state_thermal_rating()
    rating_mva = math.sqrt(3.0) * rating_amps * (row.v_nom_kV * 1e3) * 1e-6
    return rating_amps, rating_mva


# --- CACHED MAIN CALCULATION ---
@st.cache_data
def eval_weather_point(_df, Ta_C, wind_ft_s, wind_dir_from_deg, sun_time):
    """Recalculates ratings for all lines in the dataframe based on new weather."""
    out = _df.copy()
    out["WindAngleDeg"] = out["line_az_deg"].apply(lambda az: wind_angle_to_line_deg(az, wind_dir_from_deg))
    vals = out.apply(
        lambda r: calc_dynamic_rating(
            r, Ta_C=Ta_C, wind_ft_s=wind_ft_s,
            wind_angle_deg=r.WindAngleDeg, sun_time=sun_time, date_str="12 Jun"
        ),
        axis=1
    )
    out[["rating_amps", "rating_MVA_now"]] = pd.DataFrame(vals.tolist(), index=out.index)
    out["percent_now"] = 100.0 * out["p0_nominal_MVA"] / out["rating_MVA_now"]
    return out


def line_azimuth_deg(lat0, lon0, lat1, lon1):
    """Calculates the initial bearing (azimuth) from (lat0, lon0) to (lat1, lon1)."""
    φ1, φ2 = math.radians(lat0), math.radians(lat1)
    Δλ = math.radians(lon1 - lon0)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0  # 0..360


def wind_angle_to_line_deg(line_az_deg, wind_dir_from_deg):
    """Calculates the acute angle (0-90 deg) between the wind and the transmission line."""
    wind_to = (wind_dir_from_deg + 180.0) % 360.0
    diff = abs(wind_to - line_az_deg) % 360.0
    if diff > 180.0: diff = 360.0 - diff
    if diff > 90.0: diff = 180.0 - diff
    return diff  # 0..90 degrees


def make_arrow(lat, lon, direction_deg, length, head_len=0.01):
    """Creates coordinate pairs for a simple arrow."""
    lat1 = lat + length * math.cos(math.radians(direction_deg))
    lon1 = lon + length * math.sin(math.radians(direction_deg))
    lines = [{"lat0": lat, "lon0": lon, "lat1": lat1, "lon1": lon1}]
    for delta in (-25, 25):
        head_dir = direction_deg + 180 + delta
        lat2 = lat1 + (length / 4) * math.cos(math.radians(head_dir))
        lon2 = lon1 + (length / 4) * math.sin(math.radians(head_dir))
        lines.append({"lat0": lat1, "lon0": lon1, "lat1": lat2, "lon1": lon2})
    return lines


# --- Data Loading ---
@st.cache_data
def load_data(csv_path):
    """Loads and preprocesses the main line data."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: `{csv_path}` not found. Please make sure the file is in the same directory as `apps.py`.")
        st.stop()
        
    df["line_az_deg"] = df.apply(
        lambda r: line_azimuth_deg(r.lat0, r.lon0, r.lat1, r.lon1), axis=1
    )
    return df

df_main = load_data("main_df_with_coords.csv")


# --- Main Calculations ---
df_now = eval_weather_point(df_main, Ta_C=Ta, wind_ft_s=wind, wind_dir_from_deg=wind_dir_from, sun_time=sun_time)
df_now["percent_now_rounded"] = df_now["percent_now"].round(2)

def color_for(p):
    if p >= 90: return [255, 0, 0]  # Red
    if p >= 70: return [255, 215, 0] # Yellow/Gold
    return [0, 208, 132]  # Green

df_now["color"] = df_now["percent_now"].apply(color_for)


# --- Map Display (PyDeck) ---
view = pdk.ViewState(latitude=21.46, longitude=-157.99, zoom=9, pitch=0)

layer = pdk.Layer(
    "LineLayer", data=df_now,
    get_source_position=["lon0", "lat0"], get_target_position=["lon1", "lat1"],
    get_color="color", get_width=4, pickable=True,
)

arrow_len = wind / 600.0
lats = np.linspace(21, 22, 30)
lons = np.linspace(-158.5, -157.5, 30)
rows = [row for lat in lats for lon in lons for row in make_arrow(lat, lon, arrow_dir, arrow_len)]
wind_df = pd.DataFrame(rows)

wind_layer = pdk.Layer(
    "LineLayer", data=wind_df,
    get_source_position=["lon0", "lat0"], get_target_position=["lon1", "lat1"],
    get_color=[30, 144, 255], get_width=2,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer, wind_layer],
        initial_view_state=view,
        tooltip={"text": "Line {name}\nLoad: {percent_now_rounded}%"},
    )
)
st.markdown("""
**Legend:**
<span style='color:#00d084; font-size: 20px;'>●</span> <70%
<span style='color:#f1c40f; font-size: 20px;'>●</span> 70–90%
<span style='color:#e74c3c; font-size: 20px;'>●</span> >90%
""", unsafe_allow_html=True)


# --- Tabs for Detailed Analysis ---

# --- !!!!! FIX 1: REMOVED "Interactive Graphs" HEADER !!!!! ---
# The line `st.markdown("### Interactive Graphs")` has been deleted.

# --- !!!!! FIX 2: REPLACED st.tabs() WITH st.radio() !!!!! ---
# This looks like tabs but correctly remembers the active tab after a button click.
tab_options = ["Line Thresholds", "System Heatmap", "Overload Table", "AI Recommendations"]
active_tab = st.radio(
    "NavigationTabs", # A key for the radio widget
    tab_options,
    horizontal=True,
    label_visibility="collapsed" # Hides the "NavigationTabs" label
)

# This data is needed by Tab 3 AND Tab 4, and it's fast to calculate.
line_label_col = "name" if "name" in df_now.columns else "branch_name"
over_now = df_now[df_now["percent_now"] >= 100].copy()
topN = over_now.sort_values("percent_now", ascending=False).head(25).copy()


# ---------- 1) Line threshold curves ----------
if active_tab == "Line Thresholds":
    if line_label_col not in df_now.columns:
        st.info("No 'name' or 'branch_name' column found. Using first line as example.")
        pick_row = df_now.iloc[0]
        pick_label = f"Line {int(pick_row.get('original_index', 0))}"
    else:
        sorted_names = sorted(
            df_now[line_label_col].astype(str).unique(),
            key=lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x).isdigit() else 9999
        )
        pick_label = st.selectbox("Choose a line", sorted_names)
        pick_row = df_now[df_now[line_label_col].astype(str) == str(pick_label)].iloc[0]

    @st.cache_data
    def get_rating_vs_temp(_row_tuple, _wind, _wind_dir, _sun_time):
        """Calculates plot data for rating vs. temperature."""
        _row = pd.Series(_row_tuple, index=pick_row.index) 
        temps = np.linspace(0, 70, 71)
        ratings = [calc_dynamic_rating(_row, Ta_C=t, wind_ft_s=_wind, wind_angle_deg=_wind_dir, sun_time=_sun_time)[1] for t in temps]
        return temps, ratings

    @st.cache_data
    def get_rating_vs_wind(_row_tuple, _Ta, _wind_dir, _sun_time):
        """Calculates plot data for rating vs. wind."""
        _row = pd.Series(_row_tuple, index=pick_row.index)
        winds = np.linspace(0, 30, 61)
        ratings = [calc_dynamic_rating(_row, Ta_C=_Ta, wind_ft_s=w, wind_angle_deg=_wind_dir, sun_time=_sun_time)[1] for w in winds]
        return winds, ratings
    
    load_mva = float(pick_row["p0_nominal_MVA"])
    
    pick_row_tuple = tuple(pick_row)

    temps, ratings_vs_T = get_rating_vs_temp(pick_row_tuple, wind, wind_dir_from, sun_time)
    figT = go.Figure()
    figT.add_trace(go.Scatter(x=temps, y=ratings_vs_T, mode="lines", name="Rating MVA"))
    figT.add_hline(y=load_mva, line_color="gray", annotation_text="Load", annotation_position="right")
    figT.add_vline(x=Ta, line_dash="dot", annotation_text=f"Now {Ta} °C", annotation_position="top")
    figT.update_layout(
        title=f"{pick_label}: Rating vs. Temperature (at {wind:.1f} ft/s wind)",
        xaxis_title="Temperature °C", yaxis_title="MVA", height=360,
    )
    st.plotly_chart(figT, use_container_width=True)

    winds, ratings_vs_W = get_rating_vs_wind(pick_row_tuple, Ta, wind_dir_from, sun_time)
    figW = go.Figure()
    figW.add_trace(go.Scatter(x=winds, y=ratings_vs_W, mode="lines", name="Rating MVA"))
    figW.add_hline(y=load_mva, line_color="gray", annotation_text="Load", annotation_position="right")
    figW.add_vline(x=wind, line_dash="dot", annotation_text=f"Now {wind:.1f} ft/s", annotation_position="top")
    figW.update_layout(
        title=f"{pick_label}: Rating vs. Wind (at {Ta} °C)",
        xaxis_title="Wind (ft/s)", yaxis_title="MVA", height=360,
    )
    st.plotly_chart(figW, use_container_width=True)

# ---------- 2) Heatmap ----------
elif active_tab == "System Heatmap":
    @st.cache_data
    def generate_heatmap_data(_sample_df, _sun_time):
        """Performs the slow calculation for the heatmap."""
        temps_hm = np.linspace(0, 60, 25)
        winds_hm = np.linspace(0, 15, 25)
        grid = np.zeros((len(winds_hm), len(temps_hm)), dtype=int)

        for i, wv in enumerate(winds_hm):
            for j, tv in enumerate(temps_hm):
                ratings = _sample_df.apply(
                    lambda r: calc_dynamic_rating(
                        r, Ta_C=tv, wind_ft_s=wv, wind_angle_deg=r.WindAngleDeg, sun_time=_sun_time, date_str="12 Jun"
                    )[1],
                    axis=1,
                )
                grid[i, j] = int((_sample_df["p0_nominal_MVA"].to_numpy() > ratings.to_numpy()).sum())
        
        df_hm = pd.DataFrame(
            grid, index=[f"{w:.1f}" for w in winds_hm], columns=[f"{t:.1f}" for t in temps_hm],
        )
        return df_hm

    st.caption("Sweeps temperature and wind, counts how many lines would be overloaded at each point")
    sample = df_now.sample(min(len(df_now), 100), random_state=42).reset_index(drop=True)
    
    df_hm = generate_heatmap_data(sample, sun_time)
    
    figHM = px.imshow(
        df_hm,
        labels=dict(x="Temperature °C", y="Wind ft/s", color="# Overloaded (of sample)"),
        origin="lower", aspect="auto", title="Overloads Across Temperature and Wind",
    )
    figHM.add_scatter(x=[f"{Ta:.1f}"], y=[f"{wind:.1f}"], mode="markers", marker=dict(size=12, color="red", symbol="cross"), name="Current")
    st.plotly_chart(figHM, use_container_width=True)

# ---------- 3) Overload Table ----------
elif active_tab == "Overload Table":
    st.caption("Shows which lines are overloaded now, plus estimated thresholds")
    
    if len(topN):
        show_cols = [c for c in [line_label_col, "percent_now_rounded", "p0_nominal_MVA", "rating_MVA_now"] if c in topN.columns]
        st.dataframe(
            topN[show_cols].rename(columns={
                line_label_col or "": "Line",
                "percent_now_rounded": "Load %",
                "p0_nominal_MVA": "Load MVA",
                "rating_MVA_now": "Rating MVA",
            }),
            use_container_width=True,
            height=420,
        )
    else:
        st.success("No lines overloaded at current conditions")


# ---------- 4) AI Recommendations ----------
elif active_tab == "AI Recommendations":
    st.subheader("AI-Powered Grid Recommendations")

    if not GEMINI_READY:
        st.error("AI features are disabled. Check API Key or configuration.")

    elif len(topN) == 0:
        st.success("No lines are currently overloaded. No recommendations needed! ✅")
    
    else:
        st.warning("The following lines are overloaded. Click the button to get AI-powered suggestions.")
        
        st.dataframe(
            topN.rename(columns={
                line_label_col or "": "Line",
                "percent_now_rounded": "Load %",
                "p0_nominal_MVA": "Load MVA",
                "rating_MVA_now": "Rating MVA",
            })
        )
        
        if st.button("Generate Recommendations", type="primary"):
            data_string = topN.to_markdown(index=False)
            weather_string = f"Current Weather: Temperature {Ta}°C, Wind Speed {wind:.1f} ft/s, Wind From {wind_dir_from}°"
            
            system_prompt = (
                "You are an expert power grid operator and reliability analyst. "
                "Your job is to provide actionable, concise recommendations to a utility company "
                "based on a real-time list of overloaded transmission lines. "
                "Prioritize safety, grid stability, and then cost-efficiency."
            )
            
            user_prompt = (
                f"Here is the current weather on Oʻahu:\n{weather_string}\n\n"
                "The following transmission lines are overloaded. 'Load %' shows how far over capacity they are. "
                "'Load MVA' is their current power flow, and 'Rating MVA' is their current safe thermal limit.\n\n"
                f"Overload Data:\n{data_string}\n\n"
                "Please provide:\n"
                "1. A brief, 1-2 sentence summary of the current grid situation.\n"
                "2. A bulleted list of 3-5 actionable recommendations for the grid operator to manage these overloads (e.g., re-dispatching generation, curtailing load, preparing field crews)."
            )

            with st.spinner("Analyzing grid conditions and generating recommendations..."):
                try:
                    model_with_system_prompt = genai.GenerativeModel(
                        model_name="gemini-2.5-flash-preview-09-2025",
                        generation_config=generation_config,
                        system_instruction=system_prompt
                    )
                    
                    response = model_with_system_prompt.generate_content(user_prompt)
                    
                    # Store the response in session state so it persists
                    st.session_state.ai_recommendation = response.text
                    
                except Exception as e:
                    st.error(f"Error calling Gemini API: {e}")
                    if 'st.session_state.ai_recommendation' in st.session_state:
                         del st.session_state.ai_recommendation # Clear old results on error
        
        # Always display the recommendation if it exists in the session state
        # This makes it reappear after the button-click re-run
        if 'ai_recommendation' in st.session_state:

            st.markdown(st.session_state.ai_recommendation)
