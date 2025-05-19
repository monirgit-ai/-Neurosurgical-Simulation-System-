from database import init_db, log_simulation
import sqlite3

DB_FILE = "theseus_simulation.db"

def test_logging():
    print("Initializing database...")
    init_db()

    print("Inserting test log entry...")
    log_simulation(
        filename="test_brain.dcm",
        roi_area=4567,
        max_force=29.5,
        torn=True
    )

    print("Reading back the last row...")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM simulation_log ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()

    if row:
        print("✅ Log entry found:")
        print(f"  ID: {row[0]}")
        print(f"  Filename: {row[1]}")
        print(f"  ROI Area: {row[2]}")
        print(f"  Max Force: {row[3]}")
        print(f"  Torn: {'Yes' if row[4] else 'No'}")
        print(f"  Timestamp: {row[5]}")
    else:
        print("❌ No log entry found.")

if __name__ == "__main__":
    test_logging()
