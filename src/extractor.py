"""
Module 0: SVG Floor Plan Feature Extractor.
Parses SVG floor plans to extract Gross Floor Area (GFA) and room counts.
Supports geometric space group and door-scale calculation with text-based fallback.
"""

import xml.etree.ElementTree as ET
import re
import math

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


def parse_points(points_string: str) -> list:
    """Extracts (x, y) coordinates from SVG points string."""
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", points_string)
    if len(numbers) < 4:
        return []
    points = []
    for i in range(0, len(numbers) - 1, 2):
        points.append((float(numbers[i]), float(numbers[i + 1])))
    return points


def polygon_area(points: list) -> float:
    """Calculates polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def distance(p1: tuple, p2: tuple) -> float:
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def median(values: list):
    """Calculates median of list of numbers."""
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def element_label(element) -> str:
    """Combines all element attributes to form a space-separated label string."""
    label = ""
    for key, value in element.attrib.items():
        label += f" {key} {value}"
    return label


def is_polygon_or_polyline(element) -> bool:
    """Checks if XML element is a polygon or polyline (ignoring namespace)."""
    tag = element.tag.lower().split("}")[-1]
    return "polygon" in tag or "polyline" in tag


def get_points(element) -> list:
    """Retrieves point list from element's points attribute."""
    points_string = element.attrib.get("points", "")
    if not points_string:
        return []
    return parse_points(points_string)


def is_space_group(label: str) -> bool:
    """Checks if element label represents a space group."""
    parts = label.split()
    return "Space" in parts


def get_space_type(label: str):
    """Retrieves the space type word following the 'Space' keyword."""
    parts = label.split()
    if "Space" not in parts:
        return None
    index = parts.index("Space")
    if index + 1 >= len(parts):
        return "Undefined"
    return parts[index + 1]


def should_include_space(space_type: str) -> bool:
    """Checks if the space type should be included in GFA estimation."""
    if space_type is None:
        return False
    return space_type not in EXCLUDED_SPACE_LABELS


def is_door_group(label: str) -> bool:
    """Checks if element represents a door for scaling."""
    return "Door" in label.split() or "door" in label.lower()


def estimate_width_from_points(points: list):
    """Estimates door width from segment lengths of the door polygon."""
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


def extract_gfa_from_svg(svg_path: str) -> dict:
    """
    Parses SVG files containing CubiCasa groups (Space, Door) and calculates
    GFA and space counts scaled by door width.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Failed to parse SVG file: {e}")

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


def extract_features_from_svg(svg_path: str) -> dict:
    """
    Parses an SVG file and extracts geometric features.
    
    First attempts geometric space-group and door-scaling extraction.
    Falls back to text-label regex and viewBox extraction if geometric features are missing.
    
    Parameters:
        svg_path (str): Absolute path to the SVG floor plan.
        
    Returns:
        dict: A dictionary containing 'gfa' (float) and 'rooms' (int).
    """
    # 1. Try Geometric/Door-Scaled SVG extraction (CubiCasa approach)
    try:
        geo_result = extract_gfa_from_svg(svg_path)
        if geo_result is not None and geo_result["included_space_count"] > 0:
            return {
                "gfa": float(geo_result["gfa_m2"]),
                "rooms": int(geo_result["included_space_count"])
            }
    except Exception as e:
        import traceback
        print(f"Geometric extraction failed with error: {e}")
        traceback.print_exc()

    # 2. Fallback text-based and viewBox-based extraction
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Failed to parse SVG file: {e}")

    # Extract all text elements recursively
    texts = []
    for elem in root.iter():
        tag_local = elem.tag.split("}")[-1]
        if tag_local == "text":
            text_content = "".join(elem.itertext()).strip()
            if text_content:
                texts.append(text_content)

    # Extract GFA using regex on text labels
    gfa = None
    gfa_pattern = re.compile(
        r'(?:gfa|area)?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:m2|m²|sqm|sq\s*m)',
        re.IGNORECASE
    )
    for text in texts:
        match = gfa_pattern.search(text)
        if match:
            try:
                gfa = float(match.group(1))
                break
            except ValueError:
                pass

    # Extract Room Count based on text labels indicating rooms
    room_count = 0
    room_labels = [
        "bedroom", "living", "kitchen", "bathroom", "wc", "toilet",
        "hall", "corridor", "dining", "room", "lounge", "study",
        "office", "balcony", "makuuhuone", "olohuone", "keittiö",
        "kylpyhuone", "eteinen"
    ]
    for text in texts:
        text_lower = text.lower()
        if any(label in text_lower for label in room_labels) and not gfa_pattern.search(text):
            room_count += 1

    # Fallback logic for GFA based on viewport / viewBox size
    if gfa is None:
        width_str = root.attrib.get("width", "")
        height_str = root.attrib.get("height", "")
        viewbox_str = root.attrib.get("viewBox", "")

        width = None
        height = None

        if viewbox_str:
            parts = viewbox_str.split()
            if len(parts) == 4:
                try:
                    width = float(parts[2])
                    height = float(parts[3])
                except ValueError:
                    pass

        if width is None or height is None:
            clean_val = lambda s: float(re.sub(r"[^\d\.]", "", s)) if re.search(r"\d", s) else None
            try:
                if width_str:
                    width = clean_val(width_str)
                if height_str:
                    height = clean_val(height_str)
            except (ValueError, TypeError):
                pass

        if width is not None and height is not None:
            # Assume a default scaling factor (e.g. 10 pixels = 1 meter)
            # Area in m2 = (width * height) / 100
            gfa = (width * height) / 100.0
            gfa = max(30.0, min(gfa, 300.0))
        else:
            gfa = 85.0

    # Fallback logic for Room Count based on GFA
    if room_count == 0:
        room_count = max(1, int(round(gfa / 25.0)))

    return {
        "gfa": float(gfa),
        "rooms": int(room_count)
    }

