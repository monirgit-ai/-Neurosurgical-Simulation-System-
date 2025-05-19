import sqlite3
from datetime import datetime

DB_FILE = "theseus_simulation.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS simulation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        roi_area INTEGER,
        max_force REAL,
        torn INTEGER,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

def log_simulation(filename, roi_area, max_force, torn):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO simulation_log (filename, roi_area, max_force, torn, timestamp) VALUES (?, ?, ?, ?, ?)",
              (filename, roi_area, max_force, int(torn), datetime.now().isoformat()))
    conn.commit()
    conn.close()
