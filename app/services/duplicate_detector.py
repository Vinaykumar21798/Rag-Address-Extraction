from rapidfuzz import fuzz

def calculate_similarity(
    address1,
    address2
):
    return fuzz.ratio(
        address1,
        address2
    )

