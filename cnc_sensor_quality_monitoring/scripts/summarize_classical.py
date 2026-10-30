import json
from pathlib import Path

r = json.loads((Path(__file__).resolve().parents[1] / "results/metrics/classical_results.json").read_text())

for name, t in r.items():
    print("===", name.upper(), "===")
    print("n_experiments", t["n_experiments"], "class_balance", t["class_balance"])
    pb = t["process_baseline"]
    print("Process baseline RF:", pb["balanced_accuracy"], "roc", pb.get("roc_auc"), "CM", pb["confusion_matrix"])
    for m, s in t["models"].items():
        print(f"  {m:22s} bal_acc={s['balanced_accuracy']:.3f} roc={s.get('roc_auc')} f1={s['f1']:.3f}")
    rd = t.get("random_split_demo", {})
    print("Random-split demo:", rd.get("balanced_accuracy"), "roc", rd.get("roc_auc"))
    print("Ablation:")
    for g, v in t.get("ablation", {}).items():
        print(f"  {g:18s} n={v['n_features']:4d} bal_acc={v['metrics']['balanced_accuracy']:.3f}")
    print()
