import os, json, glob, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt

CSV_DIR = os.path.join(os.getcwd(), "mitdb_csv")
OUT_DIR = os.path.join(os.getcwd(), "mitdb_segments")
os.makedirs(OUT_DIR, exist_ok=True)

def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)

# simplify symbols: keep only these and map to short labels
KEEP_SYMBOLS = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',    # normal / nodal
    'V': 'V', 'E': 'V',                                 # ventricular ectopic
    'A': 'A', 'a': 'A', 'J': 'A',                       # atrial premature
    'F': 'F'                                            # fusion
}
# you can edit KEEP_SYMBOLS if you want different mapping

# load metadata
meta_path = os.path.join(CSV_DIR, "mitdb_metadata.json")
if not os.path.exists(meta_path):
    raise FileNotFoundError("mitdb_metadata.json not found in mitdb_csv. Run export first.")
with open(meta_path) as f:
    meta = {m['record']: m for m in json.load(f)}

# window params (ms)
pre_ms = 200   # ms before R-peak
post_ms = 400  # ms after R-peak

for ann_path in sorted(glob.glob(os.path.join(CSV_DIR, "*_annotations.csv"))):
    base = os.path.basename(ann_path).replace("_annotations.csv", "")
    sig_path = os.path.join(CSV_DIR, f"{base}_signals.csv")
    if not os.path.exists(sig_path):
        print("[SKIP] signals missing for", base); continue

    rec_meta = meta.get(base, None)
    if rec_meta is None:
        print("[SKIP] no metadata for", base); continue
    fs = int(rec_meta['fs'])

    df = pd.read_csv(sig_path)
    ann = pd.read_csv(ann_path)
    # single lead (lead1) — change if you want multi-lead
    sig = df['lead1'].values.astype(float)

    # filter (try; if fails, use raw)
    try:
        sigf = bandpass_filter(sig, fs)
    except Exception as e:
        print("[FILTER-ERR]", base, e); sigf = sig

    pre_samps = int(pre_ms * fs / 1000)
    post_samps = int(post_ms * fs / 1000)
    win_len = pre_samps + post_samps

    segs = []
    labs = []
    for _, row in ann.iterrows():
        sidx = int(row['sample'])
        sym = str(row['symbol']).strip()
        # map symbol to simplified label if in KEEP_SYMBOLS
        if sym not in KEEP_SYMBOLS:
            continue
        lab = KEEP_SYMBOLS[sym]
        if sidx - pre_samps < 0 or sidx + post_samps >= len(sigf):
            continue
        seg = sigf[sidx - pre_samps : sidx + post_samps]
        if len(seg) != win_len:
            continue
        segs.append(seg)
        labs.append(lab)
    if len(segs) == 0:
        print("[NO-SEGS]", base); continue
    segs = np.stack(segs).astype(np.float32)
    labs = np.array(labs)
    out_npz = os.path.join(OUT_DIR, f"{base}_segments.npz")
    np.savez_compressed(out_npz, segments=segs, labels=labs, fs=fs)
    print(f"[SEG] {base}: {segs.shape} -> {out_npz}")
