import json

with open("data/analysis/audit_failures.json", "r", encoding="utf-8") as f:
    d = json.load(f)

with open("data/analysis/audit_failures_readable.txt", "w", encoding="utf-8") as out:
    out.write(f"=== TOP FALSE AUTO-HANDLES (TOTAL: {len(d['false_auto_handles'])}) ===\n\n")
    for i, x in enumerate(d['false_auto_handles']):
        out.write(f"{i+1}. [{x['cid']}] GT: {x['intent']} | Pred: {x['pred_intent']} | Diff: {x['diff']}\n")
        out.write(f"   Msg: {x['msg']}\n")
        out.write(f"   Signals: Conf={x['signals']['intent_confidence']:.3f}, Margin={x['signals']['intent_margin']:.3f}, Sim={x['signals']['retrieval_score']:.3f}, EvCount={x['signals']['evidence_count']}\n")
        out.write(f"   Reason: {x['reason']}\n\n")

    out.write(f"\n=== TOP MISCLASSIFICATIONS (TOTAL: {len(d['misclassifications'])}) ===\n\n")
    for i, x in enumerate(d['misclassifications']):
        out.write(f"{i+1}. [{x['cid']}] GT: {x['gt']} -> Pred: {x['pred']} (Conf: {x['conf']:.3f}, Diff: {x['diff']})\n")
        out.write(f"   Msg: {x['msg']}\n\n")

    out.write(f"\n=== TOP FALSE ESCALATIONS (TOTAL: {len(d['false_escalates'])}) ===\n\n")
    for i, x in enumerate(d['false_escalates']):
        out.write(f"{i+1}. [{x['cid']}] GT: {x['intent']} | Diff: {x['diff']}\n")
        out.write(f"   Msg: {x['msg']}\n")
        out.write(f"   Reason: {x['reason']}\n\n")

print("Successfully written data/analysis/audit_failures_readable.txt")
