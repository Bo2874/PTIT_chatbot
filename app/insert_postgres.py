from conn_postgres import pg_cursor, pg_conn
import json


# Đọc file JSON
with open("../data/chunked_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Insert từng dòng
for item in data:
    pg_cursor.execute("""
        INSERT INTO informations (title, url, content)
        VALUES (%s, %s, %s)
    """, (
        item.get("title"),
        item.get("url"),
        item.get("content")
    ))

pg_conn.commit()
print("✅ Đã insert toàn bộ dữ liệu vào PostgreSQL.")

# Đóng kết nối
pg_cursor.close()
pg_conn.close()
