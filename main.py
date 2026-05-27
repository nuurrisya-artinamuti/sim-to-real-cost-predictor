#!/usr/bin/env python
"""
Main pipeline execution script.
Integrates all modules (0, 1, 2) to predict construction cost from an SVG floor plan,
target city, and target year.
"""

import sys
import os
import argparse
from src.extractor import extract_features_from_svg
from src.forecaster import MarketForecaster
from src.predictor import CostPredictor


def run_pipeline(svg_path: str, city: str, year: int, confidence: float = 90.0) -> dict:
    """Runs the complete Sim-to-Real prediction pipeline."""
    if not os.path.exists(svg_path):
        raise FileNotFoundError(f"SVG floor plan file not found at: {svg_path}")

    print(f"--- Pipeline Execution ---")
    print(f"Input SVG: {os.path.basename(svg_path)}")
    print(f"Target Location: {city}")
    print(f"Target Year: {year}")
    print(f"Confidence Level: {confidence}%")

    # Module 0: Feature Extraction
    print("\n[Step 1/3] Extracting geometric features from floor plan...")
    features = extract_features_from_svg(svg_path)
    gfa = features["gfa"]
    rooms = features["rooms"]
    print(f"  > Gross Floor Area (GFA): {gfa:.2f} m²")
    print(f"  > Room Count: {rooms} rooms")

    # Module 1: Market price forecasting
    print("\n[Step 2/3] Forecasting base market price...")
    forecaster = MarketForecaster()
    base_price = forecaster.forecast_price(city=city, year=year)
    print(f"  > Forecasted Base Market Price: {base_price:.2f} €/m²")

    # Module 2: Cost prediction
    print("\n[Step 3/3] Predicting final project construction cost...")
    predictor = CostPredictor()
    results = predictor.predict_cost(
        gfa=gfa,
        rooms=rooms,
        base_market_price=base_price,
        city=city,
        year=year,
        confidence_level=confidence
    )

    print(f"  > Expected Cost: {results['predicted_cost']:,.2f} €")
    print(f"  > {confidence}% Confidence Interval: [{results['ci_lower']:,.2f} €, {results['ci_upper']:,.2f} €]")
    print(f"--------------------------\n")

    return {
        "gfa": gfa,
        "rooms": rooms,
        "base_market_price": base_price,
        **results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict construction costs and custom Confidence Interval from an SVG floor plan."
    )
    parser.add_argument("--svg", required=True, help="Path to the SVG floor plan file.")
    parser.add_argument("--city", required=True, choices=["Helsinki", "Espoo", "Vantaa", "Tampere", "Oulu", "helsinki", "espoo", "vantaa", "tampere", "oulu"], help="Target city.")
    parser.add_argument("--year", required=True, type=int, help="Target estimation year.")
    parser.add_argument("--confidence", type=float, default=90.0, help="Confidence Interval level in percent (e.g. 90.0).")

    args = parser.parse_args()

    try:
        run_pipeline(args.svg, args.city, args.year, args.confidence)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
