import usaddress
from app.services.address_service import parse_address
from app.normalizer import normalize_address

def extract_addresses_regex(text: str):
    addresses = []
    normalized_set = set()
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    for i in range(len(lines)):
        for j in range(i + 1, min(i + 4, len(lines))):
            candidate = " ".join(lines[i:j + 1])
            try:
                parsed_tag, _ = usaddress.tag(candidate)
                if (
                    "AddressNumber" in parsed_tag
                    and "StreetName" in parsed_tag
                    and "StateName" in parsed_tag
                ):
                    parsed_dict = parse_address(candidate)
                    if parsed_dict["street"] and parsed_dict["state"]:
                        norm = normalize_address(parsed_dict)
                        norm_str = f"{norm['street']}, {norm['city']}, {norm['state']} {norm['zip']}".upper()
                        if norm_str not in normalized_set:
                            normalized_set.add(norm_str)
                            addresses.append(norm)

            except Exception:
                pass
    return {
        "addresses": addresses
    }
