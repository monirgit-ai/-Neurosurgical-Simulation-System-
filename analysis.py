import pandas as pd
import sqlite3

DB_FILE = "theseus_simulation.db"

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM simulation_log", conn)
    conn.close()
    return df

def analyze_simulations():
    df = load_data()

    if df.empty:
        print("❌ No data found in the database.")
        return

    print("📊 Statistical Summary")
    print(f"Total Simulations: {len(df)}")
    print(f"Average ROI Area: {df['roi_area'].mean():.2f} pixels")
    print(f"Average Force Applied: {df['max_force'].mean():.2f} N")
    print(f"Torn Rate: {(df['torn'].sum() / len(df)) * 100:.1f}%")
    print()

    # Optional: Export to CSV
    df.to_csv("simulation_summary.csv", index=False)
    print("✅ Exported all data to simulation_summary.csv")

    # Grouped analysis
    print("\n📁 Torn vs Intact Counts:")
    print(df['torn'].value_counts().rename({0: "Intact", 1: "Torn"}))

if __name__ == "__main__":
    analyze_simulations()
