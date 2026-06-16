ADDRESS_EXTRACTION_PROMPT = """
Extract every postal address from the document.

Return ONLY a valid JSON object.

DO NOT:
- Explain
- Apologize
- Add markdown
- Add ```json
- Add comments
- Add any text before or after JSON

The response MUST exactly match:

{{
  "addresses": [
    {{
      "street": "123 Main Street",
      "city": "Dallas",
      "state": "TX",
      "zip": "75001"
    }}
  ]
}}

If no addresses exist:

{{
  "addresses": []
}}

Document:

{document}
"""
