"""
Unit tests for the Sim-to-Real cost predictor pipeline.
To run: pytest
"""

import os
import tempfile
import pytest
from src.extractor import extract_features_from_svg
from src.forecaster import MarketForecaster
from src.predictor import CostPredictor


@pytest.fixture
def sample_svg_with_labels():
    """Creates a temporary SVG file containing text room labels and GFA."""
    content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect x="0" y="0" width="100" height="100" />
        <text x="10" y="10">Bedroom 1</text>
        <text x="10" y="30">Kitchen</text>
        <text x="10" y="50">Living Room</text>
        <text x="10" y="70">GFA: 110m2</text>
    </svg>"""
    
    with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=False) as f:
        f.write(content)
        temp_path = f.name
        
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_svg_no_labels():
    """Creates a temporary SVG file with no text labels, only viewBox."""
    content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 50">
        <rect x="0" y="0" width="80" height="50" />
    </svg>"""
    
    with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=False) as f:
        f.write(content)
        temp_path = f.name
        
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_svg_geometric():
    """Creates a temporary SVG file representing a geometric layout (CubiCasa style)."""
    content = """<svg xmlns="http://www.w3.org/2000/svg">
        <g id="space_1" class="Space Bedroom">
            <polygon points="0,0 10,0 10,10 0,10" />
        </g>
        <g id="door_1" class="Door">
            <polygon points="0,0 10,0" />
        </g>
    </svg>"""
    
    with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=False) as f:
        f.write(content)
        temp_path = f.name
        
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_extractor_with_labels(sample_svg_with_labels):
    """Tests feature extraction when text labels are present."""
    features = extract_features_from_svg(sample_svg_with_labels)
    assert features["gfa"] == 110.0
    assert features["rooms"] == 3


def test_extractor_geometric(sample_svg_geometric):
    """Tests feature extraction using geometric space/door groups."""
    features = extract_features_from_svg(sample_svg_geometric)
    # Area = 10 * 10 = 100.
    # Door width = 10. Scale = 0.90 / 10 = 0.09.
    # GFA = 100 * (0.09 ** 2) = 0.81
    assert abs(features["gfa"] - 0.81) < 1e-5
    assert features["rooms"] == 1


def test_extractor_no_labels(sample_svg_no_labels):
    """Tests feature extraction fallback logic when labels are missing."""
    features = extract_features_from_svg(sample_svg_no_labels)
    # viewBox is "0 0 80 50" -> area is 80 * 50 = 4000.
    # Default scale: area / 100 = 40.0
    assert features["gfa"] == 40.0
    # Fallback rooms: max(1, round(40 / 25)) = 2
    assert features["rooms"] == 2


def test_extractor_real_model():
    """Tests feature extraction on the real CubiCasa model.svg."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.svg")
    if not os.path.exists(model_path):
        pytest.skip("tests/model.svg not found")
    features = extract_features_from_svg(model_path)
    assert abs(features["gfa"] - 89.32) < 0.05
    assert features["rooms"] == 8


def test_forecaster():
    """Tests the market price forecaster model."""
    forecaster = MarketForecaster()
    
    # Valid forecasting
    helsinki_2026 = forecaster.forecast_price("Helsinki", 2026)
    espoo_2026 = forecaster.forecast_price("Espoo", 2026)
    vantaa_2026 = forecaster.forecast_price("Vantaa", 2026)
    tampere_2026 = forecaster.forecast_price("Tampere", 2026)
    oulu_2026 = forecaster.forecast_price("Oulu", 2026)
    
    assert isinstance(helsinki_2026, float)
    assert helsinki_2026 > 0.0
    assert isinstance(tampere_2026, float)
    assert tampere_2026 > 0.0
    assert isinstance(oulu_2026, float)
    assert oulu_2026 > 0.0
    
    # Capitalization handling
    assert forecaster.forecast_price("helsinki", 2026) == helsinki_2026
    
    # Price ordering check based on historical coefficients
    assert helsinki_2026 > espoo_2026
    assert espoo_2026 > vantaa_2026
    assert vantaa_2026 > tampere_2026
    assert tampere_2026 > oulu_2026
    
    # Invalid city checks
    with pytest.raises(ValueError):
        forecaster.forecast_price("Stockholm", 2026)


def test_predictor():
    """Tests the construction cost predictor and dynamic confidence intervals."""
    predictor = CostPredictor()
    
    # 1. Standard prediction with default 90% confidence
    results_90 = predictor.predict_cost(
        gfa=85.0,
        rooms=3,
        base_market_price=4200.0,
        city="Helsinki",
        year=2026,
        confidence_level=90.0
    )
    
    assert "predicted_cost" in results_90
    assert "ci_lower" in results_90
    assert "ci_upper" in results_90
    assert "tree_predictions" in results_90
    
    assert isinstance(results_90["predicted_cost"], float)
    assert isinstance(results_90["ci_lower"], float)
    assert isinstance(results_90["ci_upper"], float)
    
    assert results_90["ci_lower"] <= results_90["predicted_cost"]
    assert results_90["predicted_cost"] <= results_90["ci_upper"]
    assert results_90["ci_lower"] < results_90["ci_upper"]
    assert len(results_90["tree_predictions"]) == 150

    # 2. Test dynamic confidence levels: a 95% interval should be wider than a 90% interval
    results_95 = predictor.predict_cost(
        gfa=85.0,
        rooms=3,
        base_market_price=4200.0,
        city="Helsinki",
        year=2026,
        confidence_level=95.0
    )
    
    assert results_95["ci_lower"] <= results_90["ci_lower"]
    assert results_95["ci_upper"] >= results_90["ci_upper"]
