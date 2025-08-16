import json
import re

MAX_CHUNK_SIZE = 1000 # số ký tự tối đa cho mỗi chunk

def sentence_tokenize(text):
    # Tách câu theo các dấu kết thúc câu
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def custom_chunk(text, max_chunk_size=MAX_CHUNK_SIZE):
    sentences = sentence_tokenize(text)
    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:

        if current_length + len(sentence) + 1 <= max_chunk_size:
            current_chunk_sentences.append(sentence)
            current_length += len(sentence) + 1  # +1 cho khoảng trắng
        else:
            current_chunk_sentences.append(sentence)
            # Thêm chunk hiện tại vào danh sách
            chunks.append(" ".join(current_chunk_sentences).strip())

            # Bắt đầu chunk mới với overlap là câu cuối cùng
            last_sentence = current_chunk_sentences[-1]
            current_chunk_sentences = [last_sentence]
            current_length = len(last_sentence) + 1 

    # Thêm chunk cuối cùng nếu còn câu chưa xử lý
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences).strip())

    return chunks

def chunk_all_documents(metadata_path="metadata.json", output_path="chunked_data.json"):
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunked_data = []
    for item in data:
        title = item["title"]
        url = item["url"]
        content = item["content"]

        chunks = custom_chunk(content)

        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "title": title,
                "url": url,
                "chunk_id": i,
                "content": chunk
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu {len(chunked_data)} chunk vào {output_path}")

# Chạy
chunk_all_documents()
