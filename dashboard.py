"""Streamlit dashboard for Sustainable HR Tool.

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

from src import (
    data_loader,
    validation,
    preprocessing,
    features,
    race_predictor,
    utils,
)
from src import modelling  # heavy import kept separate

st.set_page_config(
    page_title="Sustainable HR Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

_MODEL_LABELS = {
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
    "ridge": "Ridge Regression",
    "linear": "Linear Regression",
}


# ──────────────────────────────────────────────────────────────────────────────
# Cached pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading and preparing data…")
def load_feature_df() -> pd.DataFrame:
    proc_folder = Path("data/processed")
    raw_folder = Path("data/raw")
    if proc_folder.exists() and any(proc_folder.glob("*.csv")):
        df = data_loader.load_all_data(str(proc_folder))
    elif raw_folder.exists() and any(raw_folder.glob("*.csv")):
        df = data_loader.load_all_data(str(raw_folder))
    else:
        df, _ = utils.generate_synthetic_and_raw()
    df_valid, _ = validation.validate_and_coerce(df)
    df_proc = preprocessing.preprocess_data(df_valid, drop_na_columns=["distance_km"])
    return features.compute_features(df_proc)


@st.cache_resource(show_spinner="Training models…")
def get_models(model_type: str):
    df = load_feature_df()
    target_hr = "avg_hr"
    target_pace = "avg_pace_min_km"
    num_cols = df.select_dtypes(include="number").columns.tolist()
    fcols_hr = [c for c in num_cols if c != target_hr and not c.startswith("predicted_")]
    fcols_pace = [c for c in num_cols if c != target_pace and not c.startswith("predicted_")]

    hr_path = Path(f"models/hr_model_{model_type}.joblib")
    pace_path = Path(f"models/pace_model_{model_type}.joblib")

    if hr_path.exists() and pace_path.exists():
        hr_model = joblib.load(hr_path)
        pace_model = joblib.load(pace_path)
    else:
        hr_path.parent.mkdir(parents=True, exist_ok=True)
        hr_model, _ = modelling.fit_sustainable_hr_model(
            df, features=fcols_hr, target=target_hr,
            model_file=str(hr_path), model_type=model_type,
        )
        pace_model, _ = modelling.fit_pace_from_hr_model(
            df, features=fcols_pace, model_file=str(pace_path), model_type=model_type,
        )

    return hr_model, pace_model, fcols_hr, fcols_pace


def _run_index(df: pd.DataFrame) -> pd.Series:
    """Return a date series if available, else a sequential run number."""
    for col in ("date", "activity_date"):
        if col in df.columns:
            try:
                return pd.to_datetime(df[col])
            except Exception:
                pass
    return pd.RangeIndex(1, len(df) + 1)


def _date_col(df: pd.DataFrame):
    """Return the name of the date column if present, else None."""
    for col in ("date", "activity_date"):
        if col in df.columns:
            return col
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Sustainable HR Tool")

    st.subheader("Data source")
    proc_folder = Path("data/processed")
    raw_folder = Path("data/raw")
    if proc_folder.exists() and any(proc_folder.glob("*.csv")):
        n = len(list(proc_folder.glob("*.csv")))
        st.success(f"data/processed ({n} file{'s' if n != 1 else ''})")
    elif raw_folder.exists() and any(raw_folder.glob("*.csv")):
        n = len(list(raw_folder.glob("*.csv")))
        st.info(f"data/raw ({n} file{'s' if n != 1 else ''})")
    else:
        st.warning("No CSV found — using synthetic data")

    if st.button("Reload data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    st.subheader("Model")
    model_type = st.selectbox(
        "Model type",
        list(_MODEL_LABELS.keys()),
        format_func=lambda k: _MODEL_LABELS[k],
        help="Switch model type — models are cached per type after first use.",
    )

    st.divider()

    st.subheader("Race prediction settings")
    target_hr_bpm = st.slider(
        "Target race HR (bpm)",
        min_value=120, max_value=200, value=155, step=1,
        help="Used for the inverse prediction: what pace can you sustain at this HR?",
    )

    st.divider()
    st.caption(
        "Drop your Garmin `Activities.csv` into `data/raw/` "
        "then click **Reload data**."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Load data & models
# ──────────────────────────────────────────────────────────────────────────────

try:
    df_full = load_feature_df()
    hr_model, pace_model, fcols_hr, fcols_pace = get_models(model_type)
except Exception as exc:
    st.error(f"Pipeline failed: {exc}")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Date-range filter (sidebar, shown only when a date column exists)
# ──────────────────────────────────────────────────────────────────────────────

dcol = _date_col(df_full)
date_from = date_to = None

if dcol:
    dates_all = pd.to_datetime(df_full[dcol], errors="coerce").dropna()
    if not dates_all.empty:
        with st.sidebar:
            st.divider()
            st.subheader("Date filter")
            min_d, max_d = dates_all.min().date(), dates_all.max().date()
            date_from = st.date_input("From", value=min_d, min_value=min_d, max_value=max_d)
            date_to   = st.date_input("To",   value=max_d, min_value=min_d, max_value=max_d)

# Apply filter
if dcol and date_from and date_to:
    mask = pd.to_datetime(df_full[dcol], errors="coerce").between(
        pd.Timestamp(date_from), pd.Timestamp(date_to)
    )
    df = df_full[mask].reset_index(drop=True)
else:
    df = df_full

x_axis = _run_index(df)
df_plot = df.copy()
df_plot["_x"] = x_axis
x_label = "Date" if hasattr(x_axis, "dtype") and str(x_axis.dtype).startswith("datetime") else "Run"


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab_overview, tab_history, tab_analysis, tab_race = st.tabs(
    ["Overview", "Training History", "Analysis", "Race Predictions"]
)


# ── Overview ──────────────────────────────────────────────────────────────────
with tab_overview:
    n_runs = len(df)
    total_dist = df["distance_km"].sum() if "distance_km" in df.columns else 0
    avg_hr_val = df["avg_hr"].mean() if "avg_hr" in df.columns else None
    avg_pace_val = df["avg_pace_min_km"].mean() if "avg_pace_min_km" in df.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", f"{n_runs:,}")
    c2.metric("Total distance", f"{total_dist:.0f} km")
    if avg_hr_val is not None:
        c3.metric("Avg HR", f"{avg_hr_val:.0f} bpm")
    if avg_pace_val is not None:
        mins = int(avg_pace_val)
        secs = int((avg_pace_val - mins) * 60)
        c4.metric("Avg pace", f"{mins}:{secs:02d} /km")

    st.subheader("Recent runs")
    show_cols = [
        c for c in ["distance_km", "duration_min", "avg_pace_min_km", "avg_hr",
                    "elevation_gain_m", "effort_type", "temperature_c"]
        if c in df.columns
    ]
    recent = df[show_cols].tail(20).iloc[::-1].reset_index(drop=True)

    try:
        selection = st.dataframe(
            recent, use_container_width=True,
            on_select="rerun", selection_mode="single-row", key="run_table",
        )
        selected_rows = selection.selection.get("rows", [])
    except Exception:
        st.dataframe(recent, use_container_width=True)
        selected_rows = []

    if selected_rows:
        run = recent.iloc[selected_rows[0]]
        with st.expander("Run detail", expanded=True):
            col_a, col_b = st.columns(2)
            for i, (k, v) in enumerate(run.items()):
                target_col = col_a if i % 2 == 0 else col_b
                if pd.api.types.is_numeric_dtype(type(v)) and pd.notna(v) and k in df.columns:
                    avg = df[k].mean()
                    delta = f"{v - avg:+.2f} vs avg"
                    target_col.metric(k, f"{v:.2f}", delta=delta)
                else:
                    target_col.metric(k, str(v) if pd.notna(v) else "—")


# ── Training History ──────────────────────────────────────────────────────────
with tab_history:
    if "distance_km" in df.columns:
        st.subheader("Distance per run")
        fig = px.bar(
            df_plot, x="_x", y="distance_km",
            labels={"_x": x_label, "distance_km": "Distance (km)"},
            color="effort_type" if "effort_type" in df.columns else None,
        )
        st.plotly_chart(fig, use_container_width=True)

    if "avg_hr" in df.columns:
        st.subheader("Heart rate over time")
        fig = px.line(
            df_plot, x="_x", y="avg_hr",
            labels={"_x": x_label, "avg_hr": "Avg HR (bpm)"},
            markers=True,
        )
        fig.update_traces(line_color="#e74c3c", marker_color="#e74c3c")
        st.plotly_chart(fig, use_container_width=True)

    if "avg_pace_min_km" in df.columns:
        st.subheader("Pace over time")
        fig = px.line(
            df_plot, x="_x", y="avg_pace_min_km",
            labels={"_x": x_label, "avg_pace_min_km": "Pace (min/km)"},
            markers=True,
        )
        fig.update_layout(yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    if "rolling_7run_load" in df.columns:
        st.subheader("Rolling 7-run training load")
        fig = px.area(
            df_plot, x="_x", y="rolling_7run_load",
            labels={"_x": x_label, "rolling_7run_load": "Cumulative distance (km)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if "ctl" in df_plot.columns and "atl" in df_plot.columns:
        st.subheader("Fitness, fatigue & form (ATL / CTL / TSB)")

        with st.expander("What are ATL, CTL, and TSB?"):
            st.markdown(
                "**ATL** (Acute Training Load) measures recent fatigue — it responds "
                "quickly to the last 1–2 weeks of training.\n\n"
                "**CTL** (Chronic Training Load) measures long-term fitness — it "
                "changes slowly over months and represents your aerobic base.\n\n"
                "**TSB** (Training Stress Balance) = CTL − ATL. "
                "A positive TSB means you are fresh and tapered; "
                "a negative TSB means you are carrying fatigue from recent hard training."
            )

        fig = px.line(
            df_plot, x="_x", y=["ctl", "atl"],
            labels={"_x": x_label, "value": "Load", "variable": "Metric"},
            color_discrete_map={"ctl": "#2ecc71", "atl": "#e74c3c"},
        )
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        if "tsb" in df_plot.columns:
            fig = px.bar(
                df_plot, x="_x", y="tsb",
                labels={"_x": x_label, "tsb": "Form (TSB)"},
                color="tsb",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ── Analysis ──────────────────────────────────────────────────────────────────
with tab_analysis:
    col1, col2 = st.columns(2)

    with col1:
        if "avg_pace_min_km" in df.columns and "avg_hr" in df.columns:
            st.subheader("HR vs Pace")
            fig = px.scatter(
                df, x="avg_pace_min_km", y="avg_hr",
                color="effort_type" if "effort_type" in df.columns else None,
                labels={"avg_pace_min_km": "Pace (min/km)", "avg_hr": "Avg HR (bpm)"},
                opacity=0.75,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "distance_km" in df.columns and "avg_hr" in df.columns:
            st.subheader("HR vs Distance")
            fig = px.scatter(
                df, x="distance_km", y="avg_hr",
                color="effort_type" if "effort_type" in df.columns else None,
                labels={"distance_km": "Distance (km)", "avg_hr": "Avg HR (bpm)"},
                opacity=0.75,
            )
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "effort_type" in df.columns:
            st.subheader("Effort breakdown")
            counts = df["effort_type"].value_counts().reset_index()
            counts.columns = ["effort_type", "count"]
            fig = px.pie(counts, names="effort_type", values="count", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "elevation_gain_m" in df.columns and "avg_pace_min_km" in df.columns:
            st.subheader("Pace vs Elevation gain")
            fig = px.scatter(
                df, x="elevation_gain_m", y="avg_pace_min_km",
                labels={"elevation_gain_m": "Elevation gain (m)", "avg_pace_min_km": "Pace (min/km)"},
                opacity=0.75,
            )
            fig.update_layout(yaxis_autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

    if "avg_hr" in df.columns and "avg_pace_min_km" in df.columns:
        st.subheader("Distance distribution by effort")
        if "effort_type" in df.columns:
            fig = px.box(
                df, x="effort_type", y="distance_km",
                labels={"effort_type": "Effort type", "distance_km": "Distance (km)"},
                color="effort_type",
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Race Predictions ──────────────────────────────────────────────────────────
with tab_race:
    st.subheader("Race prediction report")
    st.markdown(
        f"Predictions are based on your training data patterns. "
        f"**Target HR = {target_hr_bpm} bpm** (change in sidebar to update). "
        f"Model: **{_MODEL_LABELS[model_type]}**."
    )

    try:
        report = race_predictor.race_report(
            hr_model, pace_model, df,
            feature_cols_hr=fcols_hr,
            feature_cols_pace=fcols_pace,
            target_hr=float(target_hr_bpm),
        )
        st.dataframe(report, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download predictions as CSV",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="race_predictions.csv",
            mime="text/csv",
        )

        # Bar chart: predicted HR at median pace per race
        if "predicted_hr_bpm" in report.columns:
            fig = px.bar(
                report, x="race", y="predicted_hr_bpm",
                labels={"race": "Race", "predicted_hr_bpm": "Predicted HR (bpm)"},
                title="Predicted HR at median training pace",
                color="race",
            )
            fig.add_hline(y=target_hr_bpm, line_dash="dash", line_color="red",
                          annotation_text=f"Target HR ({target_hr_bpm} bpm)")
            st.plotly_chart(fig, use_container_width=True)

        # Bar chart: sustainable pace per race at target HR
        if "sustainable_pace_at_target_hr" in report.columns:
            fig = px.bar(
                report, x="race", y="sustainable_pace_at_target_hr",
                labels={"race": "Race", "sustainable_pace_at_target_hr": "Pace (min/km)"},
                title=f"Sustainable pace at {target_hr_bpm} bpm target HR",
                color="race",
            )
            fig.update_layout(yaxis_autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.warning(f"Race report could not be generated: {exc}")

    st.divider()
    st.subheader("Target finish time calculator")

    tc1, tc2, tc3 = st.columns([2, 1, 1])
    with tc1:
        selected_race = st.selectbox(
            "Race distance",
            list(race_predictor.RACE_DISTANCES_KM.keys()),
            format_func=lambda r: race_predictor.RACE_LABELS[r],
        )
    with tc2:
        t_hours = st.number_input("Hours", min_value=0, max_value=8, value=1, step=1)
    with tc3:
        t_mins = st.number_input("Minutes", min_value=0, max_value=59, value=45, step=1)

    target_total_min = float(t_hours * 60 + t_mins)
    dist_km = race_predictor.RACE_DISTANCES_KM[selected_race]

    if target_total_min > 0:
        required_pace = target_total_min / dist_km
        pm = int(required_pace)
        ps = int((required_pace - pm) * 60)
        r1, r2, r3 = st.columns(3)
        r1.metric("Target finish time", race_predictor.format_finish_time(target_total_min))
        r2.metric("Required pace", f"{pm}:{ps:02d} /km")
        r3.metric("Distance", f"{dist_km} km")

        try:
            hr_prediction = race_predictor.predict_hr_for_race(
                hr_model, df, race=selected_race,
                pace_min_km=required_pace,
                feature_cols=fcols_hr,
            )
            hr_val = hr_prediction["predicted_hr_bpm"]
            delta_hr = hr_val - target_hr_bpm
            st.metric(
                "Predicted HR at this pace",
                f"{hr_val} bpm",
                delta=f"{delta_hr:+.0f} vs target",
                delta_color="inverse",
            )
            if hr_val > target_hr_bpm + 5:
                st.warning(
                    f"This pace may push your HR above target ({target_hr_bpm} bpm). "
                    "Consider a slower pace or building more base fitness."
                )
            elif hr_val < target_hr_bpm - 5:
                st.success("This pace looks comfortable — you may have room to go faster.")
            else:
                st.info("This pace aligns well with your target HR.")
        except Exception as exc:
            st.caption(f"HR prediction unavailable: {exc}")
