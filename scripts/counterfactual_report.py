"""Aggregate paired counterfactual A/B records into per-arm environment evidence."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARM_RUNS = {
    "min": "spatial_v5_h10",
    "mean": "spatial_v5_mean",
    "prop": "spatial_v5_prop",
    "meanprop": "spatial_v5_meanprop",
}


def sign_test_p(discordant, wins):
    n = discordant
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_pairs(root, arm):
    records = []
    for path in sorted(root.glob(f"{ARM_RUNS[arm]}/cf_{arm}_t*/episodes.jsonl")):
        with path.open() as f:
            records += [json.loads(line) for line in f if line.strip()]
    usable = [r for r in records if r.get("success") is not None]
    grouped = {}
    for r in usable:
        grouped.setdefault((r["task"], r["episode"], r["state_index"], r["repeat"]), {})[
            r["alpha"]
        ] = r
    pairs = [g for g in grouped.values() if 0.0 in g and max(g) > 0.0]
    return records, usable, pairs


def arm_report(root, arm):
    records, usable, pairs = load_pairs(root, arm)
    if not pairs:
        return dict(arm=arm, pairs=0, records=len(records), usable=len(usable))
    alpha = max(max(g) for g in pairs)
    deltas = [float(g[alpha]["success"]) - float(g[0.0]["success"]) for g in pairs]
    gains = [float(g[alpha].get("q_gain", 0.0)) for g in pairs]
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / max(n - 1, 1)
    se = math.sqrt(var / n)
    wins = sum(1 for x in deltas if x > 0)
    losses = sum(1 for x in deltas if x < 0)
    if var > 0 and len(set(gains)) > 1:
        gm = sum(gains) / n
        cov = sum((g - gm) * (d - mean) for g, d in zip(gains, deltas)) / n
        sd_gain = math.sqrt(sum((g - gm) ** 2 for g in gains) / n)
        correlation = cov / (sd_gain * math.sqrt(var))
    else:
        correlation = 0.0
    per_task = {}
    for task in sorted({g[0.0]["task"] for g in pairs}):
        rows = [
            float(g[alpha]["success"]) - float(g[0.0]["success"])
            for g in pairs
            if g[0.0]["task"] == task
        ]
        per_task[str(task)] = dict(pairs=len(rows), delta=sum(rows) / len(rows))
    return dict(
        arm=arm,
        alpha=alpha,
        records=len(records),
        usable=len(usable),
        pairs=n,
        success_control=sum(float(g[0.0]["success"]) for g in pairs) / n,
        success_guided=sum(float(g[alpha]["success"]) for g in pairs) / n,
        delta=mean,
        delta_ci95=[mean - 1.96 * se, mean + 1.96 * se],
        wins=wins,
        losses=losses,
        ties=n - wins - losses,
        sign_test_p=sign_test_p(wins + losses, wins),
        q_gain_mean=sum(gains) / n,
        q_gain_vs_delta_correlation=correlation,
        displacement_mean=sum(float(g[alpha].get("displacement", 0.0)) for g in pairs) / n,
        accept_rate_mean=sum(float(g[alpha].get("accept_rate", 0.0)) for g in pairs) / n,
        max_state_error=max(float(g[a]["state_error"]) for g in pairs for a in g),
        per_task=per_task,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARM_RUNS))
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args()
    root = Path(args.artifacts)
    unknown = [a for a in args.arms if a not in ARM_RUNS]
    if unknown:
        p.error(f"unknown arms: {unknown}")
    report = {a: arm_report(root, a) for a in args.arms}
    for arm, r in report.items():
        audit = root / ARM_RUNS[arm] / "critic_gradient_audit_full.json"
        if audit.exists():
            data = json.loads(audit.read_text())
            r["gradient_audit"] = {
                k: data[k]
                for k in (
                    "head_gradient_cosine_mean",
                    "mean_gradient_direction_concentration",
                )
                if k in data
            }
    dest = root / "counterfactual_report.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    for arm, r in report.items():
        if not r.get("pairs"):
            print(f"{arm}: no usable pairs ({r.get('records', 0)} records)")
            continue
        print(
            f"{arm:>9} α={r['alpha']:<5} pairs={r['pairs']:<4} "
            f"control={r['success_control']:.3f} guided={r['success_guided']:.3f} "
            f"Δ={r['delta']:+.3f} [{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}] "
            f"W/L/T={r['wins']}/{r['losses']}/{r['ties']} p={r['sign_test_p']:.3f} "
            f"corr(q_gain,Δ)={r['q_gain_vs_delta_correlation']:+.3f} "
            f"|Δa|={r['displacement_mean']:.4f} max_state_err={r['max_state_error']:.1e}"
        )
    print(f"report: {dest}")


if __name__ == "__main__":
    main()
