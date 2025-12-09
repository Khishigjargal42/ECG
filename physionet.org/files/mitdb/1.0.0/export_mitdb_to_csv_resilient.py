import wfdb, os, pandas as pd, numpy as np, json

SRC_DIR = "."
OUT_DIR = "mitdb_csv"
os.makedirs(OUT_DIR, exist_ok=True)

records_file = os.path.join(SRC_DIR, "RECORDS")
if os.path.exists(records_file):
    with open(records_file) as f:
        records = [line.strip() for line in f if line.strip()]
else:
    records = sorted(list({os.path.splitext(f)[0] for f in os.listdir(SRC_DIR) if f.endswith('.dat')}))

meta = []

def try_rdann_variants(base):
    for ext in ["atr","at_","at-","a"]:
        path = f"{base}.{ext}"
        if os.path.exists(path):
            try:
                ann = wfdb.rdann(base, ext)
                return ann, ext
            except:
                continue
    raise FileNotFoundError(f"No readable annotation for {base}")

for r in records:
    rec_path = os.path.join(SRC_DIR, r)
    try:
        record = wfdb.rdrecord(rec_path)
    except Exception as e:
        print(f"[SKIP-REC] {r}: cannot read record (.dat/.hea) — {e}")
        continue

    try:
        ann, used_ext = try_rdann_variants(rec_path)
    except Exception as e:
        print(f"[SKIP-ANN] {r}: cannot read annotation — {e}")
        continue

    signals = record.p_signal
    fs = record.fs
    n_leads = signals.shape[1]

    df = pd.DataFrame(signals, columns=[f"lead{i+1}" for i in range(n_leads)])
    df.insert(0, "sample", np.arange(len(df)))
    out_csv = os.path.join(OUT_DIR, f"{r}_signals.csv")
    df.to_csv(out_csv, index=False)

    ann_df = pd.DataFrame({"sample": ann.sample, "symbol": ann.symbol})
    out_ann = os.path.join(OUT_DIR, f"{r}_annotations.csv")
    ann_df.to_csv(out_ann, index=False)

    meta.append({"record": r, "fs": int(fs), "n_leads": n_leads, "ann_ext": used_ext})
    print(f"[OK] {r} -> {out_csv} ann({used_ext}) fs={fs}")

with open(os.path.join(OUT_DIR, "mitdb_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("DONE: Export finished.")
