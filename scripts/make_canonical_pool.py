from pathlib import Path
import shutil

ROOT = Path(r"C:\Users\PC\OneDrive\桌面\Saarland Writing Sample")
DATA = ROOT / "data_new"

SRC = DATA / "dataset_safe_with_RAWNORM_and_CLEANv1.csv"
DST = DATA / "POOL_CANON_safe_rawnorm_cleanv1.csv"

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source pool: {SRC}")

    shutil.copy2(SRC, DST)
    print("Wrote canonical pool:", DST)

if __name__ == "__main__":
    main()
