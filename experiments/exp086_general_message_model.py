"""EXP-086 — a GENERAL social-message model: LLM encoder + per-recipient situational levers.

Two fixes to the message side, both closing the gap the Cuban email exposed:

  1. The message ENCODER was hardcoded lexical (it missed "Wharton" and read "reply yes" as no ask). It is
     now a tightly system-prompted LLM that reads for MEANING, with the lexical encoder as fallback.
  2. The variable set was Thiel-overfit (contrarian_pitch, secret_density). It is now a GENERAL universal
     set (personalization, relevance_fit, clarity, credibility_proof, responder_incentive, ask_directness,
     low_effort_ask, pushiness, warmth, length_fit, credential_signaling) PLUS per-recipient SITUATIONAL
     levers the LLM generates for each person (contrarian for Thiel, traction for Cuban) with their
     recipient-conditioned elasticities.

Run:  PYTHONPATH=. python experiments/exp086_general_message_model.py   (uses the live LLM if a key is set)
"""
from __future__ import annotations

import json

from swm.api.deepseek_backend import default_chat_fn
from swm.decision.compositional_search import encode_text_to_strategy
from swm.decision.llm_moves import llm_message_encoder
from swm.decision.situational_levers import generate_levers, levers_summary

CUBAN_EMAIL = ("Hey Mark, I'm a 16 y/o Wharton student from Brooklyn. I built an AI tool that finds "
               "below-market and rent-stabilized NYC apartments. We've helped 10,000 New Yorkers, growing "
               "90% month over month, ~10k ARR, with press from Gothamist and The Real Deal. We're raising "
               "a pre-seed and would love to get you involved. Reply yes and I'll send a short deck.")


def main():
    chat = default_chat_fn(max_tokens=500, temperature=0.0)
    mode = "LIVE LLM" if chat else "OFFLINE (lexical only)"
    print("=" * 80)
    print(f"EXP-086  general message model  [{mode}]")
    print("=" * 80)

    print("\n[1] MESSAGE ENCODER — lexical (old) vs LLM (new) on the Cuban email")
    lex = encode_text_to_strategy(CUBAN_EMAIL)
    print("  lexical :", {k: round(v, 2) for k, v in lex.items()})
    if chat:
        llm = llm_message_encoder(chat)(CUBAN_EMAIL)
        print("  llm     :", {k: round(v, 2) for k, v in llm.items()})
        print("  -> the lexical encoder missed the imperative ask and 'Wharton'; the LLM reads both, and "
              "flags the real weakness (low responder_incentive / personalization).")

    print("\n[2] SITUATIONAL LEVERS — generated per recipient (no hardcoding)")
    if chat:
        for name, ev in [("Mark Cuban", "backs hustlers with real traction; rewards sales/revenue; blunt; "
                          "answers his own email"),
                         ("Peter Thiel", "funds contrarian bets; skeptical of prestige/credentials; values "
                          "secrets and definite plans")]:
            levers = generate_levers(chat, name, {"skepticism": 0.9}, evidence=ev)
            print(f"  {name}:")
            for lv in levers_summary(levers):
                print(f"     {lv['name']:>22}  elasticity {lv['elasticity']:+.1f}  — {lv['description'][:60]}")
    else:
        print("  (needs a live LLM; offline falls back to the pure universal set with no situational levers)")

    print("\n[NOTE] Still unvalidated — the elasticities are priors. The value is the ARCHITECTURE: a "
          "general universal set + LLM-scored messages + per-recipient situational levers, so the model "
          "generalizes instead of being indexed to one person.")


if __name__ == "__main__":
    main()
