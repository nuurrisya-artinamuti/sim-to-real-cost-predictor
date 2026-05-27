"""
Streamlit Web Dashboard for the "Sim-to-Real" Construction Cost Predictor.
Displays metadata, key metrics, and an interactive Plotly S-Curve cumulative distribution.
"""

import os
import tempfile
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.extractor import extract_features_from_svg
from src.forecaster import MarketForecaster
from src.predictor import CostPredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Sim-to-Real PCM Cost Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Design Styling (CSS Injection) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Apply custom font */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }

    /* Background and containers */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 20px;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Title Styling */
    .dashboard-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 0px 4px 20px rgba(99, 102, 241, 0.25);
    }
    
    .dashboard-subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Metrics display */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #F8FAFC 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-accent {
        background: linear-gradient(135deg, #34D399 0%, #059669 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }


    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 4rem;
        color: #475569;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def build_s_curve_plot(tree_predictions, predicted_cost, ci_lower, ci_upper, confidence_level):
    """Generates an interactive Plotly S-Curve (CDF) of predicted construction costs."""
    n_trees = len(tree_predictions)
    
    # Calculate cumulative probabilities (y-axis values)
    # Ranging from 1/N to N/N
    y_values = np.arange(1, n_trees + 1) / n_trees

    fig = go.Figure()

    # 1. Plot S-Curve
    fig.add_trace(go.Scatter(
        x=tree_predictions,
        y=y_values,
        mode='lines',
        name='S-Curve (CDF)',
        line=dict(color='#6366F1', width=3.5, shape='spline'),
        hovertemplate='Cost: <b>%{x:,.2f} €</b><br>Probability: <b>%{y:.1%}</b><extra></extra>'
    ))

    # 2. Add Shaded Confidence Interval Region
    fig.add_vrect(
        x0=ci_lower,
        x1=ci_upper,
        fillcolor="rgba(99, 102, 241, 0.08)",
        layer="below",
        line_width=0,
        annotation_text=f"{confidence_level:g}% Confidence Interval",
        annotation_position="bottom right",
        annotation_font=dict(color="#94A3B8", size=10)
    )

    # 3. Add Vertical Reference Lines
    lower_p = (100.0 - confidence_level) / 2.0
    upper_p = 100.0 - lower_p
    
    # Lower Percentile Line
    fig.add_vline(
        x=ci_lower,
        line_dash="dash",
        line_color="#E2E8F0",
        line_width=1.5,
        annotation_text=f"P{lower_p:g}: {ci_lower:,.0f} €",
        annotation_position="top left",
        annotation_font=dict(color="#CBD5E1", size=10)
    )
    
    # Upper Percentile Line
    fig.add_vline(
        x=ci_upper,
        line_dash="dash",
        line_color="#E2E8F0",
        line_width=1.5,
        annotation_text=f"P{upper_p:g}: {ci_upper:,.0f} €",
        annotation_position="top right",
        annotation_font=dict(color="#CBD5E1", size=10)
    )

    # Expected Cost Line
    fig.add_vline(
        x=predicted_cost,
        line_color="#10B981",
        line_width=2.5,
        annotation_text=f"Expected: {predicted_cost:,.0f} €",
        annotation_position="top left",
        annotation_font=dict(color="#34D399", size=12, weight="bold")
    )

    # Styling the layout
    fig.update_layout(
        title=dict(
            text="Cumulative Construction Cost Distribution (S-Curve)",
            font=dict(family="Outfit", size=18, color="#F8FAFC")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        xaxis=dict(
            title=dict(text="Estimated Project Cost (€)", font=dict(color="#94A3B8")),
            gridcolor='rgba(255,255,255,0.05)',
            tickformat=",.0f",
            tickfont=dict(color="#64748B")
        ),
        yaxis=dict(
            title=dict(text="Cumulative Probability (CDF)", font=dict(color="#94A3B8")),
            gridcolor='rgba(255,255,255,0.05)',
            tickformat=".0%",
            range=[-0.05, 1.05],
            tickfont=dict(color="#64748B")
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        height=450,
        showlegend=False
    )

    return fig


def main():
    # --- Top Banner ---
    st.markdown('<h1 class="dashboard-title">🏗️ Sim-to-Real Cost Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Group G31: Dynamic Construction Decision-Support & Risk Engine</p>', unsafe_allow_html=True)

    # --- Sidebar Controls ---
    st.sidebar.markdown("### 🎛️ Simulation Parameters")
    
    # 1. Floor plan selection (Upload or Demo)
    use_demo = st.sidebar.checkbox("Use Demo Floor Plan (2-Bed Apartment)", value=True)
    
    uploaded_file = None
    if not use_demo:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Floor Plan SVG",
            type=["svg"],
            help="Upload an SVG file containing geometric room outlines or labels."
        )

    # 2. Location selection
    city = st.sidebar.selectbox(
        "Target Location / City",
        options=["Helsinki", "Espoo", "Vantaa", "Tampere", "Oulu"],
        index=0,
        help="Adjusts base real estate market price levels."
    )

    # 3. Forecast Year Selection
    year = st.sidebar.slider(
        "Target Completion Year",
        min_value=2025,
        max_value=2030,
        value=2026,
        step=1,
        help="Future year for construction inflation forecasting."
    )

    # 4. Confidence level selection
    confidence = st.sidebar.slider(
        "Confidence Interval Level (%)",
        min_value=50,
        max_value=99,
        value=90,
        step=1,
        help="Adjusts the risk envelope width extracted from the Random Forest trees."
    )

    run_clicked = st.sidebar.button("⚡ Run Cost Engine", use_container_width=True)

    # Define variables to hold simulation state
    svg_path_to_use = None
    
    # Check what floor plan to use
    if use_demo:
        svg_path_to_use = "sample_apartment.svg"
    elif uploaded_file is not None:
        # Write uploaded file to temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        svg_path_to_use = temp_file_path

    # If parameters set, automatically run or wait for click
    if svg_path_to_use is not None:
        try:
            # Load models and extract
            forecaster = MarketForecaster()
            predictor = CostPredictor()
            
            # Step 1: Feature extraction
            features = extract_features_from_svg(svg_path_to_use)
            gfa = features["gfa"]
            rooms = features["rooms"]

            # Step 2: Forecast base price
            base_price = forecaster.forecast_price(city, year)

            # Step 3: Cost and CI calculation
            results = predictor.predict_cost(gfa, rooms, base_price, confidence_level=confidence)
            
            # --- Main Content Columns ---
            col_left, col_right = st.columns([1, 2], gap="large")

            with col_left:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📋 Floor Plan Summary")
                st.write("")
                
                # Metric Cards inside Card
                m_gfa_col, m_room_col = st.columns(2)
                with m_gfa_col:
                    st.markdown(f'<div class="metric-label">Gross Floor Area</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{gfa:.1f} m²</div>', unsafe_allow_html=True)
                with m_room_col:
                    st.markdown(f'<div class="metric-label">Room Count</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{rooms}</div>', unsafe_allow_html=True)

                st.write("---")
                
                m_market_col, m_target_col = st.columns(2)
                with m_market_col:
                    st.markdown(f'<div class="metric-label">Market Level ({year})</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{base_price:,.0f} €/m²</div>', unsafe_allow_html=True)
                with m_target_col:
                    st.markdown(f'<div class="metric-label">Target City</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">{city}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Total Cost & Risk Output
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🎯 Cost Predictor Outputs")
                st.write("")
                
                st.markdown(f'<div class="metric-label">Expected Project Cost</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value metric-accent">{results["predicted_cost"]:,.2f} €</div>', unsafe_allow_html=True)
                
                st.write("---")
                
                st.markdown(f'<div class="metric-label">{confidence}% Confidence Interval</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size: 1.3rem; font-weight: 600; color: #E2E8F0;">'
                    f'{results["ci_lower"]:,.2f} € — {results["ci_upper"]:,.2f} €'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="glass-card" style="padding: 10px;">', unsafe_allow_html=True)
                # Plotly Chart
                s_curve_fig = build_s_curve_plot(
                    tree_predictions=results["tree_predictions"],
                    predicted_cost=results["predicted_cost"],
                    ci_lower=results["ci_lower"],
                    ci_upper=results["ci_upper"],
                    confidence_level=confidence
                )
                st.plotly_chart(s_curve_fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Methodology details
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### ⚙️ Risk Engine Methodology")
                st.markdown(
                    f"Unlike traditional construction risk software which relies on compute-heavy Monte Carlo simulations, "
                    f"this application employs a **Random Forest Variance Estimation** technique. "
                    f"By querying the variance in predictions across all **150 decision tree estimators** inside the forest, "
                    f"we extract a mathematically sound probability distribution. \n\n"
                    f"The **{confidence}% Confidence Interval** is calculated directly by pulling the matching percentiles from "
                    f"the estimators' outputs. This yields sub-second risk profiles suitable for interactive web platforms."
                )
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error executing cost engine pipeline: {e}")
            st.info("Ensure you have run the offline model training scripts first by running: python scripts/train_market_model.py and scripts/train_cost_model.py")
    else:
        st.info("👈 Please select or upload a floor plan SVG in the sidebar to run the prediction engine.")

    # --- Footer ---
    st.markdown('<div class="footer">PCM Group G31 "Sim-to-Real" Cost Estimation Dashboard • Antigravity 2.0 Web Application</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
