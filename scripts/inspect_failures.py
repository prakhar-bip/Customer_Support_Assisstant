import json

with open("data/analysis/audit_failures.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print(f"=== TOP FALSE AUTO-HANDLES (TOTAL: {len(d['false_auto_handles'])}) ===")
for i, x in enumerate(d['false_auto_handles']):
    print(f"{i+1}. [{x['cid']}] GT: {x['intent']} | Pred: {x['pred_intent']} | Diff: {x['diff']}")
    print(f"   Msg: {x['msg']}")
    print(f"   Signals: IntentConf={x['signals']['intent_confidence']:.3f}, Margin={x['signals']['intent_margin']:.3f}, Sim={x['signals']['retrieval_score']:.3f}, EvCount={x['signals']['evidence_count']}")
    print(f"   Reason: {x['reason']}")
    print()

print(f"=== TOP MISCLASSIFICATIONS (TOTAL: {len(d['misclassifications'])}) ===")
for i, x in enumerate(d['misclassifications']):
    print(f"{i+1}. [{x['cid']}] GT: {x['gt']} -> Pred: {x['pred']} (Conf: {x['conf']:.3f}, Diff: {x['diff']})")
    print(f"   Msg: {x['msg']}")
    print()
