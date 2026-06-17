def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50
):
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            end = text_len
        else:
            # Backtrack end to nearest whitespace boundary to avoid cutting a word in half
            while end > start and not text[end].isspace():
                end -= 1
            if end == start:
                end = start + chunk_size
                if end > text_len:
                    end = text_len
                    
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        next_start = end - overlap
        if next_start <= start:
            start = end
        else:
            # Align overlap start with the next word boundary
            while next_start < end and not text[next_start].isspace():
                next_start += 1
            while next_start < end and text[next_start].isspace():
                next_start += 1
            if next_start >= end:
                start = end
            else:
                start = next_start
                
    return chunks
