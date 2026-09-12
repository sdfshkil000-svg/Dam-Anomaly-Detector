import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from google import genai
except ImportError:
    genai = None


st.set_page_config(
    page_title="AI Anomaly Detector | Hydrodam",
    page_icon="🌊",
    layout="wide",
)

PARAMETERS = [
    "reservoir_level",
    "tailwater",
    "inflow",
    "rainfall",
    "temperature",
]

LABELS = {
    "reservoir_level": "Reservoir Level",
    "tailwater": "Tailwater",
    "inflow": "Inflow",
    "rainfall": "Rainfall",
    "temperature": "Temperature",
}

UNITS = {
    "reservoir_level": "m",
    "tailwater": "m",
    "inflow": "m³/s",
    "rainfall": "mm",
    "temperature": "°C",
}


def create_demo_history(n=365, seed=42):
    """Create synthetic demo data; replace it with actual dam history."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=n,
        freq="D",
    )

    season = np.sin(np.arange(n) * 2 * np.pi / 365)

    rainfall = np.maximum(
        0,
        rng.gamma(1.2, 5, n) + 2.5 * (season + 1),
    )

    inflow = np.maximum(
        80,
        240 + 85 * season + rainfall * 8 + rng.normal(0, 22, n),
    )

    reservoir = (
        315
        + 4.5 * season
        + np.cumsum((inflow - 235) / 2500)
        + rng.normal(0, 0.35, n)
    )
    reservoir = np.clip(reservoir, 305, 325)

    tailwater = 145 + 0.012 * inflow + rng.normal(0, 0.7, n)
    temperature = 24 + 10 * season + rng.normal(0, 2, n)

    return pd.DataFrame(
        {
            "date": days,
            "reservoir_level": reservoir,
            "tailwater": tailwater,
            "inflow": inflow,
            "rainfall": rainfall,
            "temperature": temperature,
        }
    )


def clean_history(df):
    """Validate and clean uploaded historical data."""
    missing = [c for c in PARAMETERS if c not in df.columns]

    if missing:
        raise ValueError(
            "Historical CSV is missing: " + ", ".join(missing)
        )

    out = df.copy()

    for column in PARAMETERS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=PARAMETERS).reset_index(drop=True)

    if len(out) < 30:
        raise ValueError(
            "Please provide at least 30 complete historical rows."
        )

    return out


def detect_anomaly(history, current):
    """
    Combine standardized deviations with Isolation Forest.

    This is a screening model, not a dam-safety alarm.
    """
    x = history[PARAMETERS].astype(float)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    current_df = pd.DataFrame([current], columns=PARAMETERS)
    current_scaled = scaler.transform(current_df)

    # Univariate deviation from historical behavior.
    means = x.mean()
    stds = x.std(ddof=0).replace(0, np.nan).fillna(1.0)

    z = ((pd.Series(current) - means) / stds).abs()
    z_score = float(z.max())
    z_flags = z >= 3.0

    # Multivariate anomaly detection.
    contamination = min(
        max(1.0 / max(len(history), 100), 0.005),
        0.05,
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
    )
    model.fit(x_scaled)

    iso_label = int(model.predict(current_scaled)[0])
    iso_score = float(-model.score_samples(current_scaled)[0])

    # Alert if a parameter is strongly unusual, or if the multivariate
    # model flags the observation and there is at least a moderate deviation.
    anomaly = bool(
        z_flags.any()
        or (iso_label == -1 and z_score >= 2.0)
    )

    if z_score >= 4.0 or z_flags.sum() >= 2:
        severity = "High"
    elif anomaly:
        severity = "Moderate"
    else:
        severity = "Normal"

    return {
        "anomaly": anomaly,
        "severity": severity,
        "z": z,
        "z_flags": z_flags,
        "max_z": z_score,
        "iso_label": iso_label,
        "iso_score": iso_score,
    }


def gemini_explanation(current, result, history_stats):
    """Ask Gemini 2.5 Flash to explain the statistical screening result."""
    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        os.getenv("GEMINI_API_KEY", ""),
    )

    if not api_key:
        return (
            "Gemini explanation is unavailable because "
            "GEMINI_API_KEY is not configured. The statistical "
            "detector result above is still available."
        )

    if genai is None:
        return (
            "The google-genai package is not installed. The statistical "
            "detector result above is still available."
        )

    client = genai.Client(api_key=api_key)

    stats_lines = []
    for parameter in PARAMETERS:
        stats_lines.append(
            f"{LABELS[parameter]}: "
            f"current={current[parameter]:.3f} {UNITS[parameter]}, "
            f"historical mean={history_stats.loc['mean', parameter]:.3f}, "
            f"std={history_stats.loc['std', parameter]:.3f}, "
            f"|z|={result['z'][parameter]:.2f}"
        )

    prompt = f"""
You are an engineering assistant for a hydrodam monitoring dashboard.

Interpret the statistical anomaly result below. Do NOT invent measurements
and do NOT claim a confirmed physical failure.

Give a concise engineering explanation with:
1. Overall finding.
2. Parameters driving the alert, if any.
3. Plausible non-diagnostic explanations.
4. What an operator should verify next.

Use cautious language. This is a screening tool, not a safety-critical
alarm, diagnosis, or substitute for dam-safety procedures.

Overall anomaly: {result['anomaly']}
Severity: {result['severity']}
Isolation Forest flag: {result['iso_label'] == -1}

{chr(10).join(stats_lines)}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text or "Gemini returned no explanation."
    except Exception as exc:
        return f"Gemini explanation could not be generated: {exc}"


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌊 AI Anomaly Detector for Hydrodam")
st.caption(
    "Historical-data screening for reservoir level, tailwater, "
    "inflow, rainfall and temperature."
)

with st.sidebar:
    st.header("Historical data")

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=(
            "Required columns: reservoir_level, tailwater, inflow, "
            "rainfall, temperature. An optional date column is allowed."
        ),
    )

    use_demo = st.checkbox(
        "Use demo historical data",
        value=uploaded is None,
    )

    st.markdown("**Expected columns**")
    st.code(
        "reservoir_level, tailwater, inflow, rainfall, temperature",
        language="text",
    )

if uploaded is not None and not use_demo:
    try:
        history = clean_history(pd.read_csv(uploaded))
        st.success(f"Loaded {len(history):,} historical rows.")
    except Exception as exc:
        st.error(str(exc))
        st.stop()
else:
    history = create_demo_history()
    st.info(
        "Demo data is active. Upload real historical data before using "
        "this for operational decisions."
    )


st.subheader("Enter current conditions")

input_columns = st.columns(5)
defaults = {
    parameter: float(history[parameter].median())
    for parameter in PARAMETERS
}

current = {}

for column, parameter in zip(input_columns, PARAMETERS):
    current[parameter] = column.number_input(
        LABELS[parameter],
        value=defaults[parameter],
        format="%.3f",
        help=f"Unit: {UNITS[parameter]}",
    )


if st.button(
    "🔎 Detect anomaly",
    type="primary",
    use_container_width=True,
):
    result = detect_anomaly(history, current)

    st.divider()

    if result["anomaly"]:
        st.error(
            f"⚠️ Anomaly detected — {result['severity']} screening alert"
        )
    else:
        st.success(
            "✅ No significant anomaly detected in the supplied parameters"
        )

    metric_columns = st.columns(3)

    metric_columns[0].metric(
        "Max |z-score|",
        f"{result['max_z']:.2f}",
    )

    metric_columns[1].metric(
        "Isolation Forest",
        "Anomaly" if result["iso_label"] == -1 else "Normal",
    )

    metric_columns[2].metric(
        "History rows",
        f"{len(history):,}",
    )

    st.subheader("Parameter assessment")

    assessment = pd.DataFrame(
        [
            {
                "Parameter": LABELS[parameter],
                "Current": current[parameter],
                "Unit": UNITS[parameter],
                "Historical mean": history[parameter].mean(),
                "Historical min": history[parameter].min(),
                "Historical max": history[parameter].max(),
                "|z|": result["z"][parameter],
                "Flag": (
                    "⚠️ Unusual"
                    if result["z_flags"][parameter]
                    else "Normal"
                ),
            }
            for parameter in PARAMETERS
        ]
    )

    st.dataframe(
        assessment,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("AI engineering interpretation")

    history_stats = history[PARAMETERS].agg(["mean", "std"])

    with st.spinner(
        "Gemini 2.5 Flash is interpreting the screening result..."
    ):
        st.write(
            gemini_explanation(
                current,
                result,
                history_stats,
            )
        )

    st.caption(
        "Important: This tool is a screening aid. It does not replace "
        "dam-safety instrumentation, engineering review, operating "
        "procedures, or emergency systems."
    )


with st.expander("About the detector"):
    st.write(
        "The app compares the five current measurements with historical "
        "behavior using standardized deviations and an Isolation Forest. "
        "Gemini 2.5 Flash then explains the statistical result. For "
        "production use, calibrate thresholds and models using validated "
        "dam-specific data."
    )
