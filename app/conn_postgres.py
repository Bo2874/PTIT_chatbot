import psycopg2

# --- Kết nối PostgreSQL --- 
pg_conn = psycopg2.connect(
    host="localhost",
    dbname="chatbot_ptit",
    user="postgres",
    password="28072004"
)
pg_cursor = pg_conn.cursor() #Tạo một đối tượng cursor(con trỏ) để làm việc với database (truy vấn, insert, update, ...).
