import re
import usaddress
from app.logger import logger
USPS_ABBREVIATIONS = {
    'STREET': 'ST',
    'AVENUE': 'AVE',
    'BOULEVARD': 'BLVD',
    'DRIVE': 'DR',
    'ROAD': 'RD',
    'COURT': 'CT',
    'PLACE': 'PL',
    'SQUARE': 'SQ',
    'TRAIL': 'TRL',
    'CIRCLE': 'CIR',
    'LANE': 'LN',
    'TERRACE': 'TER',
    'PARKWAY': 'PKWY',
    'HIGHWAY': 'HWY',
    'CIRCLE': 'CIR',
    'SUITE': 'STE',
    'FLOOR': 'FL',
    'APARTMENT': 'APT',
    'BUILDING': 'BLDG',
    'NORTH': 'N',
    'SOUTH': 'S',
    'EAST': 'E',
    'WEST': 'W',
    'NORTHEAST': 'NE',
    'NORTHWEST': 'NW',
    'SOUTHEAST': 'SE',
    'SOUTHWEST': 'SW',
}

def normalize_address(raw_address: str) -> str:
    try:
        parsed = parse_address(raw_address)
        street = parsed.get("street") or ""
        city = parsed.get("city") or ""
        state = parsed.get("state") or ""
        zip_code = parsed.get("zip") or ""
        street = apply_usps_abbreviations(street)
        normalized = f"{street}, {city}, {state} {zip_code}".upper().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    except Exception:
        return raw_address.upper()

def apply_usps_abbreviations(text: str) -> str:
    if not text:
        return text
    text = text.upper()
    for full, abbrev in USPS_ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(full) + r'\b'
        text = re.sub(pattern, abbrev, text)
    return text

def parse_address(raw_address: str):
    try:
        parsed, _ = usaddress.tag(raw_address)
        street_parts = []
        if "AddressNumber" in parsed:
            street_parts.append(parsed["AddressNumber"])
        if "StreetName" in parsed:
            street_parts.append(parsed["StreetName"])
        if "StreetNamePostType" in parsed:
            street_parts.append(parsed["StreetNamePostType"])
        return {
            "street": " ".join(street_parts),
            "city": parsed.get("PlaceName"),
            "state": parsed.get("StateName"),
            "zip": parsed.get("ZipCode")
        }

    except Exception as e:
        logger.warning(f"Address parsing error for '{raw_address}': {str(e)}")
        return {
            "street": None,
            "city": None,
            "state": None,
            "zip": None
        }

