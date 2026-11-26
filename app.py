
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import torch

from src.predict import (
    ROOT,
    THRESHOLD,
    load_model,
    predict_exoplanet,
    normalize_light_curve
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ExoDiscover AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

EXPECTED_LENGTH = 3197


# =====================================================
# DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080d1a, #101b32);
    color: white;
}

[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #263b5c;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #ffffff;
}

div[data-testid="stMetric"] {
    background: #142139;
    border: 1px solid #263b5c;
    padding: 18px;
    border-radius: 12px;
}

.stButton button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton button:hover {
    background: #1d4ed8;
    color: white;
}

div[data-testid="stProgress"] > div > div {
    background-color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def get_model():
    return load_model()


# =====================================================
# TRANSIT REGION ESTIMATION
# =====================================================

def estimate_transit_region(flux):

    """
    Finds the lowest rolling-median region in a
    normalized light curve.

    This is only a visual signal-analysis estimate.
    It is NOT a confirmed planetary transit detector.
    """

    normalized = normalize_light_curve(
        np.asarray(flux, dtype=np.float32)
    )

    window = 41

    smooth = (
        pd.Series(normalized)
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )

    center = int(np.argmin(smooth))

    half_width = window // 2

    start = max(
        0,
        center - half_width
    )

    end = min(
        len(flux) - 1,
        center + half_width
    )

    return {
        "start": start,
        "end": end,
        "center": center,
        "normalized": normalized
    }


# =====================================================
# LIGHT CURVE FIGURE
# =====================================================

def create_light_curve_figure(
    flux,
    transit_region=None,
    cursor=None,
    title="Stellar Light Curve",
    y_title="Observed Flux"
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(flux)),
            y=flux,
            mode="lines",
            line=dict(
                color="#38bdf8",
                width=1.4
            ),
            name="Observed Stellar Flux"
        )
    )

    # Highlight possible low-flux region

    if transit_region is not None:

        fig.add_vrect(
            x0=transit_region["start"],
            x1=transit_region["end"],
            fillcolor="rgba(250, 204, 21, 0.20)",
            line_width=0,
            annotation_text="Possible Transit Region",
            annotation_position="top left"
        )

    # Moving scanner

    if cursor is not None:

        fig.add_vline(
            x=cursor,
            line_color="#facc15",
            line_width=2
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,20,40,0.7)",
        xaxis_title="Observation Index",
        yaxis_title=y_title,
        height=420,
        hovermode="x unified",
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=40
        )
    )

    return fig


# =====================================================
# LIVE SCAN ANIMATION
# =====================================================

def show_scan_animation(flux, transit_region):

    total_points = len(flux)

    chart_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    progress_bar = st.progress(0)

    scan_steps = 40

    for step in range(scan_steps + 1):

        current_index = int(
            (step / scan_steps) * (total_points - 1)
        )

        fig = create_light_curve_figure(
            flux=flux,
            transit_region=transit_region,
            cursor=current_index,
            title="LIVE STELLAR LIGHT CURVE TRACKING"
        )

        chart_placeholder.plotly_chart(
            fig,
            use_container_width=True,
            key=f"live_scan_{step}"
        )

        percentage = int(
            (step / scan_steps) * 100
        )

        progress_bar.progress(percentage)

        progress_placeholder.markdown(
            f"**Signal Tracking: {percentage}%**"
        )

        if percentage < 25:

            status = (
                "Initializing stellar signal tracking..."
            )

        elif percentage < 50:

            status = (
                "Tracking brightness variations..."
            )

        elif percentage < 75:

            status = (
                "Examining low-flux signal regions..."
            )

        elif percentage < 100:

            status = (
                "Completing signal preprocessing..."
            )

        else:

            status = (
                "Signal tracking completed."
            )

        status_placeholder.info(
            f"🔭 {status}"
        )

        time.sleep(0.06)

    progress_placeholder.success(
        "100% — Stellar light curve tracking completed."
    )

    return True


# =====================================================
# CONFUSION MATRIX
# =====================================================

def show_confusion_matrix(tn, fp, fn, tp):

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Predicted Non-Planet",
            "Predicted Exoplanet"
        ],
        y=[
            "Actual Non-Planet",
            "Actual Exoplanet"
        ],
        color_continuous_scale="Blues",
        aspect="auto"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=430,
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# BATCH PREDICTION
# =====================================================

def predict_batch(model, flux_array):

    normalized_curves = []

    for flux in flux_array:

        normalized = normalize_light_curve(
            np.asarray(flux, dtype=np.float32)
        )

        normalized_curves.append(normalized)

    X = np.stack(
        normalized_curves
    ).astype(np.float32)

    X = torch.from_numpy(X).unsqueeze(1)

    device = next(model.parameters()).device

    X = X.to(device)

    probabilities_list = []

    model.eval()

    batch_size = 64

    total_batches = int(
        np.ceil(len(X) / batch_size)
    )

    progress = st.progress(0)

    status = st.empty()

    with torch.no_grad():

        for batch_number, start in enumerate(
            range(0, len(X), batch_size),
            start=1
        ):

            batch = X[
                start:start + batch_size
            ]

            logits = model(batch)

            probabilities = torch.sigmoid(
                logits
            ).flatten()

            probabilities_list.extend(
                probabilities.cpu().numpy().tolist()
            )

            progress.progress(
                batch_number / total_batches
            )

            status.write(
                f"AI processing batch {batch_number} "
                f"of {total_batches}..."
            )

    progress.empty()
    status.empty()

    probabilities = np.asarray(
        probabilities_list
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return probabilities, predictions


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🪐 ExoDiscover AI")

    st.caption(
        "AI-Powered Exoplanet Detection"
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Detect Exoplanet",
            "Model Performance",
            "About Project"
        ]
    )

    st.divider()

    st.caption("MODEL: 1D CNN")
    st.caption("DATASET: NASA Kepler")
    st.caption("THRESHOLD: 0.747613")
    st.caption("STATUS: Ready")


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title("🪐 ExoDiscover AI")

    st.subheader(
        "Discover worlds beyond our solar system."
    )

    st.write(
        "An AI-powered exoplanet detection system "
        "that analyzes stellar light curves using "
        "a trained 1D Convolutional Neural Network."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Samples",
        "5,087"
    )

    col2.metric(
        "Test Samples",
        "570"
    )

    col3.metric(
        "Detection Model",
        "1D CNN"
    )

    st.divider()

    st.subheader("Explore ExoDiscover AI")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "**🔭 Detect Exoplanet**\n\n"
            "Upload stellar light curves and scan "
            "individual stars or an entire dataset."
        )

    with col2:

        st.info(
            "**📊 Model Performance**\n\n"
            "Explore evaluation metrics, candidate "
            "counts and confusion matrix."
        )

    st.divider()

    st.subheader("Current Test Dataset Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Predicted Exoplanets", "6")
    c2.metric("Predicted Non-Planets", "564")
    c3.metric("Correct Detections", "5")
    c4.metric("False Alarms", "1")

    st.warning(
        "These are predictions on the held-out test dataset. "
        "The system identifies potential candidates, not "
        "independently confirmed exoplanets."
    )


# =====================================================
# DETECTION PAGE
# =====================================================

elif page == "Detect Exoplanet":

    st.title("🔭 Exoplanet Detection")

    st.write(
        "Upload stellar brightness measurements "
        "and analyze them using the trained CNN."
    )

    st.info(
        "Expected format: one row per star with exactly "
        "3197 flux measurements. Optional LABEL column: "
        "1 = Non-Planet, 2 = Exoplanet."
    )

    uploaded_file = st.file_uploader(
        "Upload Light Curve CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            if df.empty:

                st.error(
                    "Uploaded CSV is empty."
                )

                st.stop()

            has_label = "LABEL" in df.columns

            if has_label:

                label_values = pd.to_numeric(
                    df["LABEL"],
                    errors="raise"
                )

                if not set(
                    label_values.unique()
                ).issubset({1, 2}):

                    st.error(
                        "LABEL must contain only 1 or 2."
                    )

                    st.stop()

                flux_df = df.drop(
                    columns=["LABEL"]
                )

            else:

                flux_df = df.copy()

            flux_df = flux_df.apply(
                pd.to_numeric,
                errors="raise"
            )

            if flux_df.shape[1] != EXPECTED_LENGTH:

                st.error(
                    f"Expected {EXPECTED_LENGTH} flux columns, "
                    f"received {flux_df.shape[1]}."
                )

                st.stop()

            flux_array = flux_df.to_numpy(
                dtype=np.float32
            )

            if not np.isfinite(flux_array).all():

                st.error(
                    "CSV contains missing or infinite values."
                )

                st.stop()

            st.success(
                f"Successfully loaded {len(df)} light curves."
            )

            st.subheader("Uploaded Dataset")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Light Curves",
                len(df)
            )

            c2.metric(
                "Measurements per Star",
                EXPECTED_LENGTH
            )

            c3.metric(
                "Dataset Labels",
                "Available" if has_label else "Unavailable"
            )

            st.divider()

            # -----------------------------------------
            # INDIVIDUAL DETECTION
            # -----------------------------------------

            st.subheader(
                "Individual Light Curve Analysis"
            )

            selected_row = st.selectbox(
                "Select Light Curve",
                options=list(range(len(df))),
                format_func=lambda x: f"Sample {x + 1}"
            )

            flux = flux_array[selected_row]

            # Estimate possible low-flux region

            transit_region = estimate_transit_region(
                flux
            )

            st.caption(
                "Yellow region: lowest rolling-median signal "
                "window. This is a visual guide, not a "
                "confirmed transit detection."
            )

            # Initial actual light curve

            initial_fig = create_light_curve_figure(
                flux=flux,
                transit_region=transit_region,
                title="Observed Stellar Light Curve",
                y_title="Raw Flux"
            )

            st.plotly_chart(
                initial_fig,
                use_container_width=True
            )

            # -----------------------------------------
            # START INDIVIDUAL ANALYSIS
            # -----------------------------------------

            if st.button(
                "🚀 Start Live Exoplanet Detection",
                use_container_width=True
            ):

                st.divider()

                st.subheader(
                    "🔭 LIVE STELLAR SIGNAL TRACKING"
                )

                # Animate real uploaded signal

                show_scan_animation(
                    flux,
                    transit_region
                )

                # -------------------------------------
                # CNN INFERENCE
                # -------------------------------------

                with st.status(
                    "Initializing trained CNN model...",
                    expanded=True
                ) as status:

                    model = get_model()

                    st.write(
                        "Trained 1D CNN loaded successfully."
                    )

                    st.write(
                        "Normalizing actual stellar signal..."
                    )

                    st.write(
                        "Running CNN inference on complete signal..."
                    )

                    result = predict_exoplanet(
                        model,
                        flux
                    )

                    status.update(
                        label="CNN analysis completed!",
                        state="complete",
                        expanded=False
                    )

                probability = result["probability"]

                # -------------------------------------
                # FINAL RESULT
                # -------------------------------------

                st.divider()

                st.subheader(
                    "🌍 FINAL AI DETECTION RESULT"
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Candidate Score",
                    f"{probability * 100:.2f}%"
                )

                col2.metric(
                    "Decision Threshold",
                    f"{result['threshold'] * 100:.2f}%"
                )

                if result["prediction"] == 1:

                    st.success(
                        "🌍 Potential Exoplanet Candidate Detected"
                    )

                else:

                    st.info(
                        "🌌 No Exoplanet Candidate Detected"
                    )

                st.caption(
                    "The candidate score is the CNN sigmoid output. "
                    "It is not a confirmed probability of planetary existence."
                )

                # -------------------------------------
                # NORMALIZED LIGHT CURVE
                # -------------------------------------

                st.subheader(
                    "Normalized Light Curve"
                )

                normalized_fig = create_light_curve_figure(
                    flux=result["normalized_flux"],
                    transit_region=transit_region,
                    title="Normalized Stellar Signal",
                    y_title="Normalized Flux"
                )

                st.plotly_chart(
                    normalized_fig,
                    use_container_width=True
                )

                # -------------------------------------
                # SIGNAL REGION DETAILS
                # -------------------------------------

                st.subheader(
                    "Signal Region Analysis"
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Lowest-Flux Region Start",
                    transit_region["start"]
                )

                b.metric(
                    "Lowest-Flux Region End",
                    transit_region["end"]
                )

                c.metric(
                    "Region Center",
                    transit_region["center"]
                )

                st.caption(
                    "These indices come from a rolling-median "
                    "heuristic. They are not CNN attention scores "
                    "or verified transit boundaries."
                )

                # -------------------------------------
                # ACTUAL LABEL COMPARISON
                # -------------------------------------

                if has_label:

                    actual_label = int(
                        df.iloc[selected_row]["LABEL"]
                    )

                    actual_class = (
                        "Exoplanet"
                        if actual_label == 2
                        else "Non-Planet"
                    )

                    predicted_class = (
                        "Exoplanet Candidate"
                        if result["prediction"] == 1
                        else "Non-Planet"
                    )

                    st.subheader(
                        "Actual Dataset Label"
                    )

                    st.write(
                        f"Actual class: **{actual_class}**"
                    )

                    st.write(
                        f"AI prediction: **{predicted_class}**"
                    )

                    if (
                        (actual_label == 2)
                        == (result["prediction"] == 1)
                    ):

                        st.success(
                            "Prediction matches the dataset label."
                        )

                    else:

                        st.warning(
                            "Prediction differs from dataset label."
                        )

                # -------------------------------------
                # DOWNLOAD REPORT
                # -------------------------------------

                report = pd.DataFrame([{

                    "sample": selected_row + 1,

                    "prediction": result["prediction"],

                    "predicted_class": (
                        "Exoplanet Candidate"
                        if result["prediction"] == 1
                        else "Non-Planet"
                    ),

                    "candidate_score": probability,

                    "threshold": result["threshold"],

                    "lowest_flux_region_start": (
                        transit_region["start"]
                    ),

                    "lowest_flux_region_end": (
                        transit_region["end"]
                    )

                }])

                st.download_button(
                    "⬇ Download Individual Prediction",
                    data=report.to_csv(index=False),
                    file_name="exodiscover_prediction.csv",
                    mime="text/csv"
                )

            # -----------------------------------------
            # BATCH DETECTION
            # -----------------------------------------

            st.divider()

            st.subheader(
                "🌌 Batch Exoplanet Detection"
            )

            st.write(
                "Scan every light curve in the uploaded CSV "
                "and calculate predicted candidate counts."
            )

            if st.button(
                "🔍 Scan Entire Dataset",
                use_container_width=True
            ):

                model = get_model()

                with st.spinner(
                    "Preparing light curves..."
                ):

                    probabilities, predictions = predict_batch(
                        model,
                        flux_array
                    )

                total = len(predictions)

                candidate_count = int(
                    np.sum(predictions == 1)
                )

                non_planet_count = (
                    total - candidate_count
                )

                st.success(
                    "Dataset scanning completed!"
                )

                st.subheader(
                    "Prediction Distribution"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Scanned",
                    total
                )

                c2.metric(
                    "🌍 Predicted Exoplanets",
                    candidate_count
                )

                c3.metric(
                    "🌌 Predicted Non-Exoplanets",
                    non_planet_count
                )

                results_df = pd.DataFrame({

                    "Sample": np.arange(
                        1,
                        total + 1
                    ),

                    "Candidate Score (%)": (
                        probabilities * 100
                    ).round(3),

                    "Prediction": [
                        "Exoplanet Candidate"
                        if value == 1
                        else "Non-Planet"
                        for value in predictions
                    ]
                })

                if has_label:

                    actual = (
                        df["LABEL"].astype(int).to_numpy() == 2
                    ).astype(int)

                    results_df["Actual Class"] = [
                        "Exoplanet" if value == 1
                        else "Non-Planet"
                        for value in actual
                    ]

                    results_df["Correct Prediction"] = (
                        predictions == actual
                    )

                    tn = int(
                        np.sum(
                            (actual == 0) &
                            (predictions == 0)
                        )
                    )

                    fp = int(
                        np.sum(
                            (actual == 0) &
                            (predictions == 1)
                        )
                    )

                    fn = int(
                        np.sum(
                            (actual == 1) &
                            (predictions == 0)
                        )
                    )

                    tp = int(
                        np.sum(
                            (actual == 1) &
                            (predictions == 1)
                        )
                    )

                    st.divider()

                    st.subheader(
                        "Actual vs Predicted"
                    )

                    a, b, c, d = st.columns(4)

                    a.metric("True Positives", tp)
                    b.metric("False Positives", fp)
                    c.metric("True Negatives", tn)
                    d.metric("False Negatives", fn)

                    show_confusion_matrix(
                        tn, fp, fn, tp
                    )

                else:

                    st.info(
                        "No LABEL column found. Predictions "
                        "are available, but correctness cannot "
                        "be verified without actual labels."
                    )

                st.divider()

                st.subheader(
                    "All Sample Predictions"
                )

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇ Download Complete Detection Report",
                    data=results_df.to_csv(index=False),
                    file_name="exodiscover_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Unable to analyze CSV: {e}"
            )


# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.caption(
        "Evaluation on the original held-out NASA Kepler test dataset"
    )

    try:

        metrics_path = (
            ROOT / "outputs" / "final_test_metrics.json"
        )

        with open(
            metrics_path,
            "r"
        ) as file:

            metrics = json.load(file)

        tn = metrics["true_negatives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        tp = metrics["true_positives"]

        # ---------------------------------------------
        # MAIN METRICS
        # ---------------------------------------------

        st.subheader(
            "Model Evaluation Metrics"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f'{metrics["accuracy"] * 100:.2f}%'
        )

        col2.metric(
            "Precision",
            f'{metrics["precision"] * 100:.2f}%'
        )

        col3.metric(
            "Recall",
            f'{metrics["recall"] * 100:.2f}%'
        )

        col4.metric(
            "F1 Score",
            f'{metrics["f1_score"] * 100:.2f}%'
        )

        st.metric(
            "PR-AUC",
            f'{metrics["pr_auc"]:.4f}'
        )

        st.divider()

        # ---------------------------------------------
        # PREDICTION COUNTS
        # ---------------------------------------------

        st.subheader(
            "🌌 Prediction Distribution"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "🌍 Predicted Exoplanets",
            tp + fp
        )

        col2.metric(
            "🌌 Predicted Non-Exoplanets",
            tn + fn
        )

        st.divider()

        # ---------------------------------------------
        # DETECTION SUMMARY
        # ---------------------------------------------

        st.subheader(
            "🔭 Test Detection Summary"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Test Samples",
            metrics["test_samples"]
        )

        c2.metric(
            "Correct Detections",
            tp
        )

        c3.metric(
            "False Alarms",
            fp
        )

        c4.metric(
            "Missed Exoplanets",
            fn
        )

        st.divider()

        # ---------------------------------------------
        # PREDICTION DISTRIBUTION CHART
        # ---------------------------------------------

        st.subheader(
            "Prediction Distribution Chart"
        )

        distribution = pd.DataFrame({

            "Class": [
                "Predicted Exoplanets",
                "Predicted Non-Exoplanets"
            ],

            "Count": [
                tp + fp,
                tn + fn
            ]

        })

        distribution_fig = px.bar(
            distribution,
            x="Class",
            y="Count",
            text="Count",
            color="Class",
            color_discrete_sequence=[
                "#38bdf8",
                "#a78bfa"
            ]
        )

        distribution_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            distribution_fig,
            use_container_width=True
        )

        st.divider()

        # ---------------------------------------------
        # CONFUSION MATRIX
        # ---------------------------------------------

        st.subheader(
            "Confusion Matrix"
        )

        show_confusion_matrix(
            tn,
            fp,
            fn,
            tp
        )

        st.markdown("""
        **How to read the Confusion Matrix:**

        - **True Negative:** Non-planets correctly identified.
        - **False Positive:** Non-planets incorrectly marked as candidates.
        - **False Negative:** Actual exoplanets missed by the model.
        - **True Positive:** Actual exoplanets correctly identified.
        """)

        st.divider()

        # ---------------------------------------------
        # INTERPRETATION
        # ---------------------------------------------

        st.subheader(
            "📈 Detection Interpretation"
        )

        st.write(
            f"Total samples analyzed: "
            f"**{metrics['test_samples']}**"
        )

        st.write(
            f"Predicted exoplanet candidates: **{tp + fp}**"
        )

        st.write(
            f"Predicted non-exoplanets: **{tn + fn}**"
        )

        st.write(
            f"Correctly detected actual exoplanets: **{tp}**"
        )

        st.write(
            f"False alarms: **{fp}**"
        )

        st.write(
            f"Missed actual exoplanets: **{fn}**"
        )

        st.divider()

        st.subheader(
            "Decision Threshold"
        )

        st.metric(
            "Frozen Threshold",
            f'{metrics["threshold"]:.6f}'
        )

        st.caption(
            "The threshold was selected using validation data "
            "and kept fixed for final test evaluation."
        )

        st.warning(
            "Only 5 actual exoplanet examples were present "
            "in this test dataset. These results are limited "
            "and do not establish confirmed astronomical discoveries."
        )

    except FileNotFoundError:

        st.warning(
            "Final test metrics file not found. "
            "Check outputs/final_test_metrics.json."
        )

    except Exception as e:

        st.error(
            f"Unable to load metrics: {e}"
        )


# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About Project":

    st.title("🌌 About ExoDiscover AI")

    st.write("""
    ExoDiscover AI is an academic machine-learning project
    designed to identify potential exoplanet candidates
    using stellar brightness measurements.

    ### Technology Stack

    - Python
    - PyTorch
    - 1D Convolutional Neural Network
    - Streamlit
    - Plotly
    - NASA Kepler-labelled dataset

    ### How It Works

    1. Upload stellar light-curve measurements.
    2. Normalize each signal using median and IQR.
    3. Estimate a possible low-flux region for visualization.
    4. Track the actual signal using the animated scanner.
    5. Pass the complete normalized signal through the trained 1D CNN.
    6. Calculate the candidate score using sigmoid.
    7. Compare the score against the frozen threshold.

    ### Scientific Note

    A highlighted low-flux region is not automatically
    a confirmed planetary transit. The CNN prediction is
    a machine-learning candidate, not an astronomical confirmation.
    """)
