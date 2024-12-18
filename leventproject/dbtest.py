import sqlite3

# Veritabanına bağlan
conn = sqlite3.connect('db.sqlite3')  # 'db.sqlite3' dosyasının yolu
cursor = conn.cursor()

# Veri ekleme fonksiyonu
def insert_text(text):
    cursor.execute("INSERT INTO text_model (texts) VALUES (?)", (text,))
    conn.commit()
    print(f"'{text}' tablonuza eklendi.")

# Verileri listeleme fonksiyonu
def fetch_texts():
    cursor.execute("SELECT * FROM text_model")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

# Örnek kullanım
insert_text("Merhaba, Dünya!")
fetch_texts()

# Bağlantıyı kapat
conn.close()
