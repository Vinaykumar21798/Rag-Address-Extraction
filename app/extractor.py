import re
from pydantic import ValidationError
from app.logger import logger
from app.llm import generate
from app.normalizer import normalize_address
from app.prompts import ADDRESS_EXTRACTION_PROMPT
from app.schemas import AddressList
from app.regex_extractor import extract_addresses_regex

def extract_json(text: str):
    match = re.search(
        r'\{.*\}',
        text,
        re.DOTALL
    )
    if not match:
        raise ValueError(
            "No JSON object found"
        )
    return match.group()

def llm_extract(document_text: str):
    prompt = ADDRESS_EXTRACTION_PROMPT.format(
        document=document_text
    )
    response = generate(
        [
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300
    )
    return response

def validate_json(response_text: str):
    clean_json = extract_json(
        response_text
    )
    return AddressList.model_validate_json(
        clean_json
    )

def extract_addresses(document_text: str):
    try:
        response = llm_extract(
            document_text
        )
        validated = validate_json(
            response
        )
        normalized_addresses = [
            normalize_address(addr)
            for addr in validated.model_dump()["addresses"]
        ]
        logger.info(
            "PATH=LLM_SUCCESS"
        )
        return {
            "status": "LLM_SUCCESS",
            "addresses": {
                "addresses": normalized_addresses
            }
        }

    except (ValidationError, ValueError) as e:
        try:
            retry_response = retry_extract(
                document_text,
                str(e)
            )
            retry_validated = validate_json(
                retry_response
            )
            normalized_addresses = [
                normalize_address(addr)
                for addr in retry_validated.model_dump()["addresses"]
            ]
            logger.info(
                "PATH=RETRY_SUCCESS"
            )
            return {
                "status": "RETRY_SUCCESS",
                "addresses": {
                    "addresses": normalized_addresses
                }
            }

        except (ValidationError, ValueError):
            fallback_result = (
                extract_addresses_regex(
                    document_text
                )
            )
            logger.info(
                "PATH=REGEX_FALLBACK"
            )
            return {
                "status": "REGEX_FALLBACK",
                "addresses": fallback_result
            }

def retry_extract(document_text: str, error_message: str):
    retry_prompt = f"""
The previous response failed validation.

Validation Error:
{error_message}

Return ONLY valid JSON.

Document:

{document_text}
"""
    response = generate(
        [
            {
                "role": "user",
                "content": retry_prompt
            }
        ],
        max_tokens=300
    )
    return response
