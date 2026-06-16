import re
import usaddress
from rapidfuzz import fuzz
from app.database.models import Address
from app.logger import logger

def extract_components(address: Address) -> tuple[str, str]:
    house_number = ""
    street_name = ""
    if address.raw_text:
        try:
            parsed, _ = usaddress.tag(address.raw_text)
            house_number = parsed.get("AddressNumber", "")
            street_name = parsed.get("StreetName", "")

        except Exception:
            pass
    if (not house_number or not street_name) and address.street:
        try:
            parsed, _ = usaddress.tag(address.street)
            house_number = parsed.get("AddressNumber", "")
            street_name = parsed.get("StreetName", "")

        except Exception:
            pass
    if (not house_number or not street_name) and address.street:
        match = re.match(r"^(\d+\S*)\s+(.*)$", address.street.strip())
        if match:
            house_number = house_number or match.group(1)
            street_name = street_name or match.group(2)

        else:
            street_name = street_name or address.street
    return house_number.strip().upper(), street_name.strip().upper()

def determine_recommendation(addr1: Address, addr2: Address) -> str:
    hn1, sn1 = extract_components(addr1)
    hn2, sn2 = extract_components(addr2)
    city1 = (addr1.city or "").strip().upper()
    city2 = (addr2.city or "").strip().upper()
    state1 = (addr1.state or "").strip().upper()
    state2 = (addr2.state or "").strip().upper()
    zip1 = (addr1.zip or "").strip()
    zip2 = (addr2.zip or "").strip()
    if hn1 and hn2 and hn1 != hn2:
        logger.debug(f"Recommend NOT_DUPLICATE: House numbers differ ({hn1} vs {hn2})")
        return "not_duplicate"
    if zip1 and zip2 and zip1 != zip2:
        logger.debug(f"Recommend NOT_DUPLICATE: ZIP codes differ ({zip1} vs {zip2})")
        return "not_duplicate"
    if city1 and city2 and city1 != city2:
        logger.debug(f"Recommend NOT_DUPLICATE: Cities differ ({city1} vs {city2})")
        return "not_duplicate"
    if state1 and state2 and state1 != state2:
        logger.debug(f"Recommend NOT_DUPLICATE: States differ ({state1} vs {state2})")
        return "not_duplicate"
    if sn1 and sn2:
        street_sim = fuzz.ratio(sn1, sn2)
        if street_sim < 50:
            logger.debug(f"Recommend NOT_DUPLICATE: Street names completely different (score {street_sim})")
            return "not_duplicate"
    if all([hn1, hn2, sn1, sn2, city1, city2, state1, state2, zip1, zip2]):
        same_hn = (hn1 == hn2)
        same_city = (city1 == city2)
        same_state = (state1 == state2)
        same_zip = (zip1 == zip2)
        street_sim = fuzz.ratio(sn1, sn2)
        nearly_identical_street = (street_sim >= 75)
        if same_hn and same_city and same_state and same_zip and nearly_identical_street:
            logger.debug(f"Recommend MERGE: Same hn/city/state/zip and high street sim ({street_sim})")
            return "merge"
    logger.debug("Recommend MANUAL_REVIEW: Default fallback rule triggered")
    return "manual_review"

