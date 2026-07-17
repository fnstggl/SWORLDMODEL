"""Demonstration 2 — single-individual reaction through the same qualitative architecture.

"How will Dana react if I tell them I can't make dinner tonight?" The target is automatically
Tier 1 (reaction_is_the_question); several qualitative hidden-state hypotheses are built from
the supplied relationship history; each particle independently reads the exact message, reacts
internally, and chooses an observable response; the distribution is the count of those choices.

    DEEPSEEK_API_KEY=… PYTHONPATH=. python experiments/qualitative_demo_individual.py
"""
from __future__ import annotations

import calendar
import json
import time as _time
from pathlib import Path

from swm.world_model_v2.individual_reaction import simulate_individual_reaction
from swm.world_model_v2.qualitative_actor import QualitativeConfig

AS_OF = float(calendar.timegm(_time.strptime("2026-07-17", "%Y-%m-%d")))
RESULTS = Path("experiments/results")


def main():
    from swm.api.deepseek_backend import deepseek_chat_fn
    llm = deepseek_chat_fn(temperature=0.9, max_tokens=2000)
    hypo = deepseek_chat_fn(temperature=0.8, max_tokens=2000)
    t0 = _time.time()
    result = simulate_individual_reaction(
        person_id="Dana",
        stimulus="Hey — I'm really sorry, work exploded and I can't make dinner tonight. "
                 "Can we do next week instead? I'll book somewhere good.",
        context={
            "relationship": "close friend of eight years",
            "role": "close friend", "your_role": "close friend",
            "history": [
                "You two have a standing monthly dinner you both protect",
                "You cancelled on Dana twice last month, both times for work",
                "Dana cooked an elaborate dinner at their place last time",
                "Dana mentioned feeling like work always wins with you",
                "Dana helped you through a rough stretch last winter",
            ],
            "goals": ["keep the friendship close", "feel valued, not squeezed in"],
        },
        llm=llm, n_hypotheses=3, samples_per_hypothesis=3, seed=17, as_of=AS_OF,
        config=QualitativeConfig(llm=llm, hypothesis_llm=hypo, n_hypotheses=3,
                                 max_llm_calls=42))
    result["wall_s"] = round(_time.time() - t0, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "qualitative_demo_individual.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    print("SAMPLES:")
    for s in result["samples"]:
        print(f"  [{s['hypothesis_id'][:32]:32s}] {s['observable_response']:12s} "
              f"| feels: {str(s['internal_reaction'])[:70]}")
    print(f"\nRAW DISTRIBUTION: {json.dumps(result['raw_qualitative_simulation_distribution'])}")
    print(f"CALIBRATION: {result['calibration_status']}")
    print(f"llm_calls={result['llm_calls']} wall={result['wall_s']}s\nwrote {path}")


if __name__ == "__main__":
    main()
