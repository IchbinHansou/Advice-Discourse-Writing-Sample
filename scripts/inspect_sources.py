import json
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root = parent of /scripts

def main():
    cfg_path = ROOT / "data_new" / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find run_config.json at: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    ra_path = ROOT / cfg["paths"]["ra"]
    conf_path = ROOT / cfg["paths"]["confession"]
    aita_path = ROOT / cfg["paths"]["aita"]

    print("Project ROOT:", ROOT)
    print("Config path:", cfg_path)
    print("RA path:", ra_path)
    print("Confession path:", conf_path)
    print("AITA path:", aita_path)

    print("\n=== Rela Advice CSV columns ===")
    ra = pd.read_csv(ra_path, nrows=5)
    print(list(ra.columns))

    print("\n=== Confessions columns ===")
    conf = pd.read_csv(conf_path, nrows=5)
    print(list(conf.columns))

    print("\n=== AITA sqlite tables ===")
    con = sqlite3.connect(str(aita_path))
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    print(tables)

    for t in tables[:10]:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t});").fetchall()]
        print(f"\n--- {t} columns ---")
        print(cols)

    con.close()

if __name__ == "__main__":
    main()