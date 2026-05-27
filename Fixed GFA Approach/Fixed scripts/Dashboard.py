import re
import math
import tempfile
import pandas as pd
import streamlit as st
import joblib
from xml.etree import ElementTree as ET


# ============================================================
# FILE PATHS — CHANGE IF NEEDED
# ============================================================

MODEL_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\final_linear_cost_model.pkl"
MAE_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\final_model_mae.txt"

PRICE_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\Fixed datasets\New Input 2 - Cities Finland.xlsx"
PRICE_SHEET = "City - Year - Price"


# ============================================================
# SETTINGS
# ============================================================

STANDARD_DOOR_WIDTH_M = 0.90

EXCLUDED_SPACE_LABELS = [
    "Outdoor",
    "Garage",
    "CarPort",
    "SwimmingPool",
    "Elevator",
    "OpenToBelow",
    "Garbage",
    "RetailSpace"
]


# ============================================================
# BASIC GEOMETRY FUNCTIONS
# ============================================================

def parse_points(points_string):
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", points_string)

    if len(numbers) < 4:
        return []

    points = []

    for i in range(0, len(numbers) - 1, 2):
        points.append((float(numbers[i]), float(numbers[i + 1])))

    return points


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    area = 0.0

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def distance(p1, p2):
    return math.sqrt(
        (p2[0] - p1[0]) ** 2 +
        (p2[1] - p1[1]) ** 2
    )


def median(values):
    if not values:
        return None

    values = sorted(values)
    n = len(values)
    mid = n // 2

    if n % 2 == 1:
        return values[mid]

    return (values[mid - 1] + values[mid]) / 2


# ============================================================
# SVG HELPERS
# ============================================================

def element_label(element):
    label = ""

    for key, value in element.attrib.items():
        label += f" {key} {value}"

    return label


def is_polygon_or_polyline(element):
    tag = element.tag.lower()
    return "polygon" in tag or "polyline" in tag


def get_points(element):
    points_string = element.attrib.get("points", "")

    if not points_string:
        return []

    return parse_points(points_string)


def is_space_group(label):
    parts = label.split()
    return "Space" in parts


def get_space_type(label):
    parts = label.split()

    if "Space" not in parts:
        return None

    index = parts.index("Space")

    if index + 1 >= len(parts):
        return "Undefined"

    return parts[index + 1]


def should_include_space(space_type):
    if space_type is None:
        return False

    return space_type not in EXCLUDED_SPACE_LABELS


def is_door_group(label):
    return "Door" in label.split() or "door" in label.lower()


def estimate_width_from_points(points):
    if len(points) < 2:
        return None

    lengths = []

    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        d = distance(p1, p2)

        if d > 0:
            lengths.append(d)

    if not lengths:
        return None

    lengths = sorted(lengths)

    return lengths[-1]


# ============================================================
# EXTRACT GFA FROM SVG
# ============================================================

def extract_gfa_from_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    total_gfa_svg = 0.0
    included_space_count = 0
    excluded_space_count = 0
    undefined_space_count = 0
    door_widths = []

    def walk(element):
        nonlocal total_gfa_svg
        nonlocal included_space_count
        nonlocal excluded_space_count
        nonlocal undefined_space_count
        nonlocal door_widths

        label = element_label(element)

        if is_space_group(label):
            space_type = get_space_type(label)
            group_area = 0.0

            for child in element.iter():
                if is_polygon_or_polyline(child):
                    points = get_points(child)

                    if points:
                        group_area += polygon_area(points)

            if group_area > 0:
                if should_include_space(space_type):
                    total_gfa_svg += group_area
                    included_space_count += 1

                    if space_type == "Undefined":
                        undefined_space_count += 1
                else:
                    excluded_space_count += 1

            return

        if is_door_group(label):
            for child in element.iter():
                if is_polygon_or_polyline(child):
                    points = get_points(child)

                    if points:
                        width = estimate_width_from_points(points)

                        if width is not None and width > 0:
                            door_widths.append(width)

            return

        for child in list(element):
            walk(child)

    walk(root)

    median_door_width_svg = median(door_widths)

    if median_door_width_svg is None or median_door_width_svg <= 0:
        return None

    scale_m_per_svg_unit = STANDARD_DOOR_WIDTH_M / median_door_width_svg
    gfa_m2 = total_gfa_svg * (scale_m_per_svg_unit ** 2)

    return {
        "gfa_m2": gfa_m2,
        "included_space_count": included_space_count,
        "excluded_space_count": excluded_space_count,
        "undefined_space_count": undefined_space_count,
        "median_door_width_svg": median_door_width_svg,
        "scale_m_per_svg_unit": scale_m_per_svg_unit
    }


# ============================================================
# LOAD MODEL, MAE, AND PRICE DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_prices():
    prices = pd.read_excel(
        PRICE_FILE,
        sheet_name=PRICE_SHEET
    )

    prices.columns = prices.columns.str.strip()

    prices = prices.rename(columns={
        "City": "city",
        "Year": "year",
        "Price per square meter (EUR/m2)": "price_m2"
    })

    return prices


def load_mae():
    with open(MAE_FILE, "r") as f:
        return float(f.read())


model = load_model()
prices = load_prices()
mae = load_mae()


# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Construction Cost Prediction Tool",
    layout="wide"
)

st.title("🏠 Construction Cost Prediction Tool")
st.write(
    "Upload a CubiCasa `model.svg`, choose a Finnish city and year, "
    "and receive an ML-based construction cost estimate."
)

uploaded_svg = st.file_uploader(
    "Upload CubiCasa model.svg",
    type=["svg"]
)

city_options = sorted(prices["city"].dropna().unique())
year_options = sorted(prices["year"].dropna().unique())

city = st.selectbox("Select city", city_options)
year = st.selectbox("Select year", year_options)

if uploaded_svg is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_file:
        temp_file.write(uploaded_svg.read())
        temp_svg_path = temp_file.name

    extracted = extract_gfa_from_svg(temp_svg_path)

    if extracted is None:
        st.error("Could not calculate GFA. Door scale could not be detected.")
    else:
        gfa_m2 = extracted["gfa_m2"]
        rooms = extracted["included_space_count"]

        price_row = prices[
            (prices["city"] == city) &
            (prices["year"] == year)
        ]

        if price_row.empty:
            st.error("No price data available for this city and year.")
        else:
            price_m2 = float(price_row["price_m2"].iloc[0])
            base_cost = gfa_m2 * price_m2
            complexity_factor = 1 + (rooms * 0.01)

            input_data = pd.DataFrame([{
                "gfa_m2": gfa_m2,
                "rooms": rooms,
                "city": city,
                "year": year,
                "price_m2": price_m2,
                "base_cost": base_cost,
                "complexity_factor": complexity_factor
            }])

            predicted_cost = model.predict(input_data)[0]

            lower_bound = max(predicted_cost - mae, 0)
            upper_bound = predicted_cost + mae

            st.subheader("Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Gross Floor Area", f"{gfa_m2:,.1f} m²")
            col2.metric("Price per m²", f"€{price_m2:,.0f}")
            col3.metric("Predicted Cost", f"€{predicted_cost:,.0f}")

            st.subheader("ML-Based Risk Range")
            st.info(
                f"Based on the model MAE (€{mae:,.0f}), "
                f"the typical prediction range is approximately "
                f"€{lower_bound:,.0f} – €{upper_bound:,.0f}."
            )

            st.subheader("Extracted Floor Plan Features")

            feature_table = pd.DataFrame({
                "Feature": [
                    "GFA",
                    "Included spaces",
                    "Excluded spaces",
                    "Undefined spaces",
                    "Median door width in SVG units",
                    "Scale factor"
                ],
                "Value": [
                    f"{gfa_m2:,.2f} m²",
                    extracted["included_space_count"],
                    extracted["excluded_space_count"],
                    extracted["undefined_space_count"],
                    f"{extracted['median_door_width_svg']:.2f}",
                    f"{extracted['scale_m_per_svg_unit']:.6f} m/SVG unit"
                ]
            })

            st.table(feature_table)

            st.subheader("Model Input Data")
            st.dataframe(input_data)

            st.subheader("Cost Breakdown")

            breakdown = pd.DataFrame({
                "Metric": [
                    "Base Cost = GFA × price/m²",
                    "Complexity Factor",
                    "Predicted Cost",
                    "Typical Lower Bound",
                    "Typical Upper Bound"
                ],
                "Value": [
                    f"€{base_cost:,.0f}",
                    f"{complexity_factor:.2f}",
                    f"€{predicted_cost:,.0f}",
                    f"€{lower_bound:,.0f}",
                    f"€{upper_bound:,.0f}"
                ]
            })

            st.table(breakdown)

else:
    st.info("Upload a CubiCasa SVG file to start.")