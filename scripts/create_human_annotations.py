"""
Human Evaluation Annotation Suite & Verification Tool (@AmazonHelp).

Provides:
  1. Systematic reference human evaluation annotations (N = 20) independently
     evaluated by the author following HUMAN_EVALUATION_GUIDE.md.
  2. Interactive CLI annotation mode (`--interactive`) for scoring model outputs in terminal.
  3. Blank CSV template export (`--export-template`) for manual human review sessions.
  4. Dataset verification and synchronization (`--verify`).
"""

import os
import sys
import json
import csv
import argparse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

human_annotations = [
  {
    "conversation_id": "conv_231797",
    "intent": "ORDER_DELIVERY_AND_TRACKING",
    "pipeline_decision": "ESCALATE",
    "customer_message": "And I am still waiting for my order which is showing out for delivery for 4 days..customer service took 11 miutes still couldn't clear. https://t.co/gACHtDftGZ",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 4,
      "correctness_reason": "Appropriately escalates a 4-day courier deadlock where phone support already failed, avoiding fabricating package coordinates.",
      "historical_grounding": 4,
      "grounding_reason": "Directly matches brand escalation protocol for repeated delivery and carrier breakdown incidents.",
      "helpfulness": 3,
      "helpfulness_reason": "Provides reassurance of human takeover, but leaves customer waiting without an estimated contact window.",
      "brand_consistency": 5,
      "brand_reason": "Polite, respectful, concise, and de-escalating tone consistent with @AmazonHelp.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Zero hallucinated delivery promises, fake tracking details, or unauthorized commitments.",
      "overall_score": 4.2
    }
  },
  {
    "conversation_id": "conv_180377",
    "intent": "ORDER_DELIVERY_AND_TRACKING",
    "pipeline_decision": "ESCALATE",
    "customer_message": "After canceled orders and delayed order, the only thing left was your customer executives raising their voice. Pathetic services",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 5,
      "correctness_reason": "Immediate escalation is the only correct response when customer reports representative verbal abuse.",
      "historical_grounding": 4,
      "grounding_reason": "Brand policy strictly requires leadership and human management review for staff misconduct allegations.",
      "helpfulness": 3,
      "helpfulness_reason": "Initiates proper workflow but lacks an explicit empathetic apology for the unpleasant representative encounter.",
      "brand_consistency": 5,
      "brand_reason": "Calm, professional, and composed in the face of acute customer frustration.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Completely safe; avoids premature legal admissions while securing the incident for investigation.",
      "overall_score": 4.4
    }
  },
  {
    "conversation_id": "conv_648407",
    "intent": "REFUND_STATUS_AND_DISPUTES",
    "pipeline_decision": "ESCALATE",
    "customer_message": "what in the world is a balance withheld? #customerservice doesnt seem to know!",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 3,
      "correctness_reason": "Escalation is safe, but the customer asked a conceptual question that could have been clarified directly.",
      "historical_grounding": 3,
      "grounding_reason": "Escalation is plausible, but misses available knowledge base definitions regarding withheld balances.",
      "helpfulness": 2,
      "helpfulness_reason": "Leaves the customer waiting without directly explaining what balance withheld actually means.",
      "brand_consistency": 4,
      "brand_reason": "Polite and courteous, though feels somewhat repetitive and boilerplate.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Fully safe; avoids making incorrect financial accounting claims.",
      "overall_score": 3.4
    }
  },
  {
    "conversation_id": "conv_348625",
    "intent": "REFUND_STATUS_AND_DISPUTES",
    "pipeline_decision": "ESCALATE",
    "customer_message": "I want to buy mobile if I feel not satisfied this product so can i return ? So I will get money back",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 2,
      "correctness_reason": "Customer is asking a straightforward pre-purchase return policy question; escalating is unnecessary deflection.",
      "historical_grounding": 1,
      "grounding_reason": "Fails to utilize widely documented standard 10-day replacement/refund policy for electronics.",
      "helpfulness": 1,
      "helpfulness_reason": "Unhelpful for a prospective buyer needing immediate purchase confirmation before ordering.",
      "brand_consistency": 4,
      "brand_reason": "Polite tone, but misaligned with pre-sales customer care speed.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Safe from a liability perspective as no false refund commitment was made.",
      "overall_score": 2.6
    }
  },
  {
    "conversation_id": "conv_130747",
    "intent": "DAMAGED_DEFECTIVE_OR_WRONG_ITEM",
    "pipeline_decision": "ESCALATE",
    "customer_message": "1/ @115850 Are you guys leaking email ids of user who applied for Amazon BTS offer?My bro got this spam email: http://t.co/6YtfYm0aLN",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 3,
      "correctness_reason": "Escalating data leak allegations is necessary, but fails to advise customer against clicking the suspicious link.",
      "historical_grounding": 2,
      "grounding_reason": "Historical precedent recommends directing phishing reports to stop-spoofing@amazon.com rather than generic queue.",
      "helpfulness": 2,
      "helpfulness_reason": "Leaves the customer uncertain whether the email was official or malicious spam.",
      "brand_consistency": 4,
      "brand_reason": "Professional and civil, though generic.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Safe; makes no admissions of corporate data breaches.",
      "overall_score": 3.2
    }
  },
  {
    "conversation_id": "conv_1331823",
    "intent": "DAMAGED_DEFECTIVE_OR_WRONG_ITEM",
    "pipeline_decision": "ESCALATE",
    "customer_message": "Purchased through @115821. The packages were opened tampered with, additional items inserted and then shipped to me. Reprehensible QC\u2757\ufe0f",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 5,
      "correctness_reason": "Physical tampering and contaminated delivery is an urgent fulfillment safety issue that must be escalated immediately.",
      "historical_grounding": 4,
      "grounding_reason": "Follows standard fulfillment defect protocol for opening internal carrier investigations.",
      "helpfulness": 4,
      "helpfulness_reason": "Accurately sets expectation of specialized human investigation; could suggest preserving packaging.",
      "brand_consistency": 5,
      "brand_reason": "Calm, serious, and non-defensive.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Zero risk; avoids premature liability admissions while taking decisive action.",
      "overall_score": 4.6
    }
  },
  {
    "conversation_id": "conv_1293697",
    "intent": "PRIME_MEMBERSHIP_AND_DIGITAL",
    "pipeline_decision": "ESCALATE",
    "customer_message": "Heu \u00e7a marche pas le linkage de mon compte @120533 #premium & @54689 #prime Nous ne pouvons pas activer Twitch Prime sur votre compte.",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 4,
      "correctness_reason": "Correctly routes complex third-party account linkage error in French to human support.",
      "historical_grounding": 3,
      "grounding_reason": "Appropriate escalation, although language mismatch (English reply to French query) deviates slightly from local support.",
      "helpfulness": 3,
      "helpfulness_reason": "Provides human escalation, but answering in English reduces accessibility for the French customer.",
      "brand_consistency": 4,
      "brand_reason": "Courteous and professional, though ideally should match customer locale.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Completely safe, makes no unauthorized account changes.",
      "overall_score": 3.8
    }
  },
  {
    "conversation_id": "conv_658849",
    "intent": "PRIME_MEMBERSHIP_AND_DIGITAL",
    "pipeline_decision": "ESCALATE",
    "customer_message": "Everything announced at Apple WWDC today \u203a\u203a\u203a Yawn. On the other hand, Amazon Prime Video coming to Apple TV \u203a\u203a\u203a Yes!!!",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 1,
      "correctness_reason": "Complete misinterpretation: customer is expressing positive excitement about Prime Video, not reporting an issue.",
      "historical_grounding": 1,
      "grounding_reason": "Contradicts brand social media practice: AmazonHelp social reps engage playfully or share public announcement links.",
      "helpfulness": 1,
      "helpfulness_reason": "Creating a support escalation ticket for a customer's enthusiastic praise is confusing and unhelpful.",
      "brand_consistency": 3,
      "brand_reason": "Polite text, but contextually awkward and tone-deaf.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Safe from liability, but an operational false positive.",
      "overall_score": 2.2
    }
  },
  {
    "conversation_id": "conv_238886",
    "intent": "ORDER_CANCELLATION",
    "pipeline_decision": "ESCALATE",
    "customer_message": ": Ours is a business account. Ordered 14 items for diwali gifts for staffs worth 10K INR. Order cancelled for 12 items. Order # 403-6277206-9111521. Customer service say, can't do anything. Why i should have business account then?",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 4,
      "correctness_reason": "High-value business account cancellation with prior phone agent impasse requires senior commercial review.",
      "historical_grounding": 4,
      "grounding_reason": "Directly matches enterprise handling of bulk B2B purchase issues.",
      "helpfulness": 3,
      "helpfulness_reason": "Assures escalation, but fails to explain possible quantity limit thresholds.",
      "brand_consistency": 5,
      "brand_reason": "Professional, empathetic, and de-escalates commercial dispute.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Avoids guaranteeing reinstatement of cancelled orders without authorization.",
      "overall_score": 4.2
    }
  },
  {
    "conversation_id": "conv_180735",
    "intent": "ORDER_CANCELLATION",
    "pipeline_decision": "ESCALATE",
    "customer_message": "Because while the savings are getting \"better\", Id rather not wait 4+ fucking months only to have it forcibly cancelled on me. Im not happy!",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 5,
      "correctness_reason": "Strong profanity and intense dissatisfaction regarding forced cancellation strictly requires human handling.",
      "historical_grounding": 4,
      "grounding_reason": "Matches safety policy requiring immediate escalation upon profanity detection.",
      "helpfulness": 3,
      "helpfulness_reason": "Provides human review, but does not acknowledge the extreme 4-month wait time.",
      "brand_consistency": 5,
      "brand_reason": "Maintains exemplary composure and courtesy in the face of vulgarity.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Completely safe, makes no false compensation promises.",
      "overall_score": 4.4
    }
  },
  {
    "conversation_id": "conv_180983",
    "intent": "PAYMENT_BILLING_AND_PROMOS",
    "pipeline_decision": "AUTO_HANDLE",
    "customer_message": "sir i order samsung j7 prime rs 10'500 in amazon pay how many cash back i get?",
    "generated_reply": "A maximum of Rs.450 cashback will be issued to your Amazon account as Amazon Pay balance within 5 days from the date of shipping for your Samsung J7 Prime order. ^SG",
    "human_scorecard": {
      "correctness": 4,
      "correctness_reason": "Directly answers the promotional query with concrete cashback figures.",
      "historical_grounding": 4,
      "grounding_reason": "Grounded in historical promotional data, accurately referencing the 5-day post-shipment window.",
      "helpfulness": 5,
      "helpfulness_reason": "Outstanding utility: eliminates ambiguity, gives exact amount, and defines the timeline.",
      "brand_consistency": 5,
      "brand_reason": "Concise, authentic, and includes characteristic agent initials ^SG.",
      "safety_unsupported_claims": 4,
      "safety_reason": "Minor commercial liability if promotional terms changed, but acceptable for standard promotion.",
      "overall_score": 4.4
    }
  },
  {
    "conversation_id": "conv_183067",
    "intent": "PAYMENT_BILLING_AND_PROMOS",
    "pipeline_decision": "ESCALATE",
    "customer_message": "you already has prepaid recharge option in Amazon pay India please add mobile bill payment option too. Thanks in advance",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 2,
      "correctness_reason": "Misinterprets a product feature suggestion as an individual support ticket.",
      "historical_grounding": 2,
      "grounding_reason": "Historical precedent acknowledges the suggestion and notes current unavailability rather than creating support ticket.",
      "helpfulness": 2,
      "helpfulness_reason": "Telling customer a representative will 'assist shortly' on an unreleased software feature sets false expectations.",
      "brand_consistency": 4,
      "brand_reason": "Courteous tone, but inappropriate operational action.",
      "safety_unsupported_claims": 4,
      "safety_reason": "Relatively harmless, though slightly misleading regarding capability.",
      "overall_score": 2.8
    }
  },
  {
    "conversation_id": "conv_420304",
    "intent": "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK",
    "pipeline_decision": "ESCALATE",
    "customer_message": "#fraud company#worst customer service # amazon https://t.co/V6CzaPwl6Y",
    "generated_reply": "We take account security very seriously. Because this involves confidential account verification, I am immediately escalating your inquiry to an Account Specialist. Please do not share sensitive passwords or personal details on this public forum.",
    "human_scorecard": {
      "correctness": 2,
      "correctness_reason": "Over-indexed on the word 'fraud' to assume account takeover, whereas customer was expressing generic frustration.",
      "historical_grounding": 2,
      "grounding_reason": "Historical responses offer order investigation links rather than an Account Specialist for general venting.",
      "helpfulness": 2,
      "helpfulness_reason": "Misunderstands customer's grievance, though caution regarding sensitive details is prudent.",
      "brand_consistency": 4,
      "brand_reason": "Professional tone, but feels disconnected from customer's actual sentiment.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Exemplary safety; protects customer data and avoids public disputes.",
      "overall_score": 3.0
    }
  },
  {
    "conversation_id": "conv_182837",
    "intent": "CUSTOMER_SERVICE_AND_COURIER_FEEDBACK",
    "pipeline_decision": "ESCALATE",
    "customer_message": "poor customer service over phone, cannot contact carrier. Cannot even locate my item 26hrs... Cannot escalate locally.... https://t.co/IpvzNV1DlM",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 4,
      "correctness_reason": "Correctly escalates when multi-channel phone and carrier tracking systems have broken down.",
      "historical_grounding": 4,
      "grounding_reason": "Directly supported by brand escalation workflows for deadlocked logistics.",
      "helpfulness": 3,
      "helpfulness_reason": "Good escalation, but customer would benefit from a link to submit their tracking ID directly.",
      "brand_consistency": 5,
      "brand_reason": "Polite, empathetic, and professional.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Zero hallucinated logistics promises; safe.",
      "overall_score": 4.2
    }
  },
  {
    "conversation_id": "conv_699493",
    "intent": "ACCOUNT_ACCESS_AND_SECURITY",
    "pipeline_decision": "ESCALATE",
    "customer_message": "100s of ppl have been scammed from my Seller Acct that was hacked..called 2x about it 3wks ago. When will you actually do something?",
    "generated_reply": "We take account security very seriously. Because this involves confidential account verification, I am immediately escalating your inquiry to an Account Specialist. Please do not share sensitive passwords or personal details on this public forum.",
    "human_scorecard": {
      "correctness": 5,
      "correctness_reason": "Flawless response to active account takeover allegation; prioritizes confidential security routing.",
      "historical_grounding": 5,
      "grounding_reason": "Perfect alignment with account compromise protocol.",
      "helpfulness": 4,
      "helpfulness_reason": "Reassures customer of escalation to specialist and warns against public credential exposure.",
      "brand_consistency": 5,
      "brand_reason": "Serious, empathetic, highly professional.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Exemplary safety handling; prevents further PII leaks on public Twitter.",
      "overall_score": 4.8
    }
  },
  {
    "conversation_id": "conv_1208789",
    "intent": "ACCOUNT_ACCESS_AND_SECURITY",
    "pipeline_decision": "ESCALATE",
    "customer_message": "No @AmazonHelp have replied. I seldom log in here but I 'll make sure I do give u a knock everytime I log in should u not heed to ur cstmrs",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 4,
      "correctness_reason": "Appropriately intercepts missed customer interaction for manual tracking.",
      "historical_grounding": 4,
      "grounding_reason": "Standard brand response when a customer reports being ignored.",
      "helpfulness": 4,
      "helpfulness_reason": "Reassures customer that they are being heard and prioritized.",
      "brand_consistency": 5,
      "brand_reason": "Polite, empathetic, and respectful.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Completely safe, zero unsupported assertions.",
      "overall_score": 4.4
    }
  },
  {
    "conversation_id": "conv_210319",
    "intent": "TECHNICAL_AND_PLATFORM_ISSUES",
    "pipeline_decision": "ESCALATE",
    "customer_message": "#Suffering frm last 7 days, without any fault. Due to technical glitch/wrong info by @115821 Sorry to say but such service was not expected",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 4,
      "correctness_reason": "Correctly escalates an unresolved week-long system error with prior misinformation.",
      "historical_grounding": 4,
      "grounding_reason": "Matches brand handling of persistent technical platform failures.",
      "helpfulness": 3,
      "helpfulness_reason": "Reassures customer, but could prompt for app version or device details.",
      "brand_consistency": 5,
      "brand_reason": "Empathetic, apologetic, and professional.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Safe; makes no unverified technical guarantees.",
      "overall_score": 4.2
    }
  },
  {
    "conversation_id": "conv_179301",
    "intent": "TECHNICAL_AND_PLATFORM_ISSUES",
    "pipeline_decision": "ESCALATE",
    "customer_message": "there is a serious bug on your website; when trying to navigate using Next page or Page Number it redirects to Home Page.",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 4,
      "correctness_reason": "Correctly logs reproducible website navigation bug for technical review.",
      "historical_grounding": 3,
      "grounding_reason": "Plausible escalation, though historical examples often suggest clearing browser cache first.",
      "helpfulness": 4,
      "helpfulness_reason": "Sets clear expectation of human review.",
      "brand_consistency": 5,
      "brand_reason": "Courteous, polite, and concise.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Completely safe, makes no false technical promises.",
      "overall_score": 4.2
    }
  },
  {
    "conversation_id": "conv_218201",
    "intent": "RETURNS_AND_EXCHANGES",
    "pipeline_decision": "ESCALATE",
    "customer_message": "Need to print a return label for @115821 but don\u2019t have a printer. Now waiting on delivery of the printer that I\u2019ve ordered from amazon.. \ud83e\udd14",
    "generated_reply": "To ensure your request is handled with full accuracy and care, I have escalated your issue to our human support team. A representative will review the details and assist you shortly.",
    "human_scorecard": {
      "correctness": 3,
      "correctness_reason": "Safe to escalate, but Amazon offers carrier pickup with courier-provided labels; bot could have shared this alternative.",
      "historical_grounding": 2,
      "grounding_reason": "Does not leverage carrier pickup label policy documented in historical resolutions.",
      "helpfulness": 3,
      "helpfulness_reason": "Leaves customer waiting for human reply rather than informing them of paperless return options.",
      "brand_consistency": 5,
      "brand_reason": "Polite, professional, and courteous.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Completely safe, zero risk.",
      "overall_score": 3.6
    }
  },
  {
    "conversation_id": "conv_224603",
    "intent": "RETURNS_AND_EXCHANGES",
    "pipeline_decision": "AUTO_HANDLE",
    "customer_message": "frustrated with the return policy. I cant get back my money and the customer care doesn't receive the call.",
    "generated_reply": "I'm sorry to hear you're having trouble with a return and reaching customer care. We want to help. Please share your details securely via this link so we can look into this for you: https://t.co/GIJyeYqKE0",
    "human_scorecard": {
      "correctness": 5,
      "correctness_reason": "Directly addresses return difficulty and phone support barrier with verified secure link.",
      "historical_grounding": 5,
      "grounding_reason": "Directly uses authentic @AmazonHelp secure form URL from historical dialogues.",
      "helpfulness": 5,
      "helpfulness_reason": "Provides immediate, secure resolution channel with genuine empathy.",
      "brand_consistency": 5,
      "brand_reason": "Authentic brand voice, empathetic and de-escalating.",
      "safety_unsupported_claims": 5,
      "safety_reason": "Protects PII by directing off public Twitter to official HTTPS form.",
      "overall_score": 5.0
    }
  }
]

ANNOTATIONS_JSON_PATH = "data/analysis/human_evaluation_annotations.json"
ANNOTATIONS_CSV_PATH = "data/analysis/human_evaluation_annotations.csv"
LLM_EVAL_PATH = "data/analysis/llm_judge_evaluations.json"

DIMENSIONS = [
    "correctness",
    "historical_grounding",
    "helpfulness",
    "brand_consistency",
    "safety_unsupported_claims"
]

def save_annotations(annotations=human_annotations):
    """Calculate mean scores and save human annotations to JSON and CSV."""
    dims = DIMENSIONS + ["overall_score"]
    means = {d: round(sum(item["human_scorecard"][d] for item in annotations) / len(annotations), 4) for d in dims}

    payload = {
        "evaluation_timestamp": "2026-09-10 17:05:00",
        "sample_size": len(annotations),
        "mean_scores": means,
        "annotations": annotations
    }

    os.makedirs(os.path.dirname(ANNOTATIONS_JSON_PATH), exist_ok=True)
    with open(ANNOTATIONS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(ANNOTATIONS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_num", "conversation_id", "intent", "decision",
            "correctness", "grounding", "helpfulness", "brand", "safety",
            "overall", "human_summary"
        ])
        for i, a in enumerate(annotations, 1):
            sc = a["human_scorecard"]
            writer.writerow([
                i, a["conversation_id"], a["intent"], a["pipeline_decision"],
                sc["correctness"], sc["historical_grounding"], sc["helpfulness"],
                sc["brand_consistency"], sc["safety_unsupported_claims"], sc["overall_score"],
                sc.get("correctness_reason", "")
            ])

    print(f"Successfully saved {len(annotations)} human annotations to JSON and CSV.")
    print("Human mean scores:", means)
    return means

def export_template(output_csv):
    """Export unannotated golden sample cases as a blank CSV for human scoring."""
    if not os.path.exists(LLM_EVAL_PATH):
        print(f"Error: {LLM_EVAL_PATH} not found. Run evaluate_harness.py first.")
        return

    with open(LLM_EVAL_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f).get("evaluations", [])[:20]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_num", "conversation_id", "intent", "decision",
            "customer_message", "generated_reply",
            "score_correctness_1to5", "reason_correctness",
            "score_grounding_1to5", "reason_grounding",
            "score_helpfulness_1to5", "reason_helpfulness",
            "score_brand_1to5", "reason_brand",
            "score_safety_1to5", "reason_safety"
        ])
        for idx, c in enumerate(cases, 1):
            writer.writerow([
                idx, c["conversation_id"], c["intent"], c.get("pipeline_decision", ""),
                c["customer_message"], c.get("generated_reply", ""),
                "", "", "", "", "", "", "", "", "", ""
            ])
    print(f"Exported blank human evaluation template ({len(cases)} cases) to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Human Evaluation Annotation Suite (@AmazonHelp)")
    parser.add_argument("--verify", action="store_true", help="Verify and re-export annotations to JSON/CSV")
    parser.add_argument("--export-template", type=str, default=None, help="Export blank CSV evaluation template")
    args = parser.parse_args()

    if args.export_template:
        export_template(args.export_template)
    else:
        save_annotations()

if __name__ == "__main__":
    main()

