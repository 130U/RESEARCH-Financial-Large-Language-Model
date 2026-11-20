"""Recompute every figure in the article and compare it with the text.

    python reproduce/run_all.py

Evaluates the closed forms in src/finllm, reads the Trial's measurements from
evidence/trial.json, runs the contract checks on the Trial model on the CPU,
writes reproduce/results.md and exits with status 1 if any figure disagrees.
Needs PyTorch; about half a minute.
"""

from __future__ import annotations

import json
import pathlib
import sys
from fractions import Fraction
from math import exp, log

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reproduce"))

import torch  # noqa: E402

import contract_checks  # noqa: E402
from finllm import accounting, architecture  # noqa: E402
from finllm import permission_math as pm  # noqa: E402
from finllm.objective import family_view_weights, presence_probability  # noqa: E402
from finllm.permissions import Permission, pair_loss  # noqa: E402


def three_class_consistency(strength: float) -> dict:
    """Minimize expected cross-entropy plus the consistency term over free three-class logits."""
    p = torch.tensor([[0.6, 0.3, 0.1], [0.85, 0.1, 0.05]], dtype=torch.float64)
    logits = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([logits], lr=1.0, max_iter=4000, tolerance_grad=1e-14, tolerance_change=1e-16,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        anchors = -(p * logits.log_softmax(-1)).sum()
        terms = pair_loss(logits[0:1], logits[1:2], torch.tensor([0]), torch.tensor([0]), [Permission.EQUIVALENT])
        loss = anchors + strength * 0.5 * terms["consistency"].sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    q = logits.detach().softmax(-1)
    return {"errors_cancel": float(((q[0] - p[0]) + (q[1] - p[1])).abs().max()),
            "distance_to_average": float((q - p.mean(0)).abs().max())}


def main() -> int:
    sections: list[tuple[str, list]] = []
    rows: list = []

    def section(title: str) -> None:
        nonlocal rows
        rows = []
        sections.append((title, rows))

    def check(label, value, article, where, tol=0, shown=None):
        rows.append((label, value, shown if shown is not None else article, where, abs(value - article) <= tol))

    def check_true(label, ok, where, shown="holds"):
        rows.append((label, None, shown, where, bool(ok)))

    trial = json.loads((ROOT / "evidence" / "trial.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ Section 3
    section("Label-preserving is not distribution-preserving (Section 3)")
    weight = pm.evidence_weight(0.70, 0.90)
    check("Evidence an edit carries when a calibrated reader moves from 0.70 to 0.90 (nats)", weight, 1.35, "§3", tol=0.005)
    check("The same as a likelihood ratio", exp(weight), 3.86, "§3", tol=0.005)
    table = {0.03: (0.728, 0.872, 0.028), 0.1: (0.756, 0.844, 0.056), 0.3: (0.779, 0.821, 0.079),
             1: (0.793, 0.807, 0.093), 3: (0.797, 0.803, 0.097), 10: (0.799, 0.801, 0.099)}
    for strength, (original, edited, error) in table.items():
        r = pm.unlicensed_consistency_optimum(0.70, 0.90, strength)
        check(f"Strength {strength:g}: prediction at the original", r["original"], original, "§3", tol=0.0005)
        check(f"Strength {strength:g}: prediction at the edited text", r["edited"], edited, "§3", tol=0.0005)
        check(f"Strength {strength:g}: calibration error at each end", r["error"], error, "§3", tol=0.0005)
        check_true(f"Strength {strength:g}: the two errors are equal and opposite",
                   abs(r["original"] + r["edited"] - 1.60) < 1e-9, "§3, eq. 3.2")
    check("Limit of the error as the strength grows", pm.consistency_limit_error(0.70, 0.90), 0.100, "§3, eq. 3.2", tol=1e-9)
    check("Both predictions in the limit", pm.unlicensed_consistency_optimum(0.70, 0.90, 1e7)["original"], 0.800, "§3", tol=1e-5)
    check_true("A true equivalence costs nothing at any strength",
               all(pm.unlicensed_consistency_optimum(0.8, 0.8, s)["error"] == 0 for s in (0.1, 1, 10, 1e6)), "§3")
    errors = [pm.unlicensed_consistency_optimum(0.70, 0.90, s)["error"] for s in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100)]
    check_true("The error increases with the strength and stays below half the gap",
               all(a < b for a, b in zip(errors, errors[1:])) and errors[-1] < 0.1, "§3, eq. 3.2")
    for strength in (0.3, 1, 3):
        closed = pm.unlicensed_consistency_optimum(0.70, 0.90, strength)
        fitted = contract_checks.consistency_fixed_point(0.70, 0.90, strength)
        check_true(f"Minimizing the implemented loss over free logits reaches the same predictions (strength {strength:g})",
                   abs(fitted["original"] - closed["original"]) < 1e-6 and abs(fitted["edited"] - closed["edited"]) < 1e-6, "§3")
    three = three_class_consistency(50.0)
    check_true("Three classes: the error vectors at the two ends are equal and opposite", three["errors_cancel"] < 1e-7, "§3")
    check_true("Three classes: at high strength both predictions approach the average posterior",
               three["distance_to_average"] < 0.01, "§3")
    check("Safe margin when labels are assigned at posterior odds of at least 2 (nats)", pm.safe_margin(2), 1.39, "§3, eq. 3.3", tol=0.005)
    d_star = pm.calibrated_response(0.6, 0.3, 0.3, 0.6)
    check_true("A calibrated model at odds of exactly 2 at both ends meets that margin with zero penalty",
               abs(d_star - pm.safe_margin(2)) < 1e-12 and pm.response_penalty(pm.safe_margin(2), d_star) < 1e-12, "§3, eq. 3.3")
    check_true("A model that does not respond pays the full margin",
               abs(pm.response_penalty(pm.safe_margin(2), 0.0) - pm.safe_margin(2)) < 1e-12, "§3")

    # ------------------------------------------------------------------ Section 4
    section("One update, one objective (Section 4)")
    share = Fraction(trial["review"]["split"]["train"], trial["review"]["task_pool_with_a_literal_target"])
    check("Training examples in the task pool", trial["review"]["split"]["train"], 705, "§4")
    check("Task pool", trial["review"]["task_pool_with_a_literal_target"], 1008, "§4")
    for views, article in ((2, 69.9), (3, 91.0), (5, 99.2)):
        check(f"Families of {views} views: chance that a held-out view has a sibling in training (%)",
              float(100 * accounting.sibling_in_training(views, share)), article, "§4", tol=0.05)
    weights = family_view_weights(["edited"] * 21 + ["plain"])
    check_true("A family with twenty edits weighs as much as a family with none", sum(weights[:21]) == weights[21] == Fraction(1, 2), "§4")
    adaptation = architecture.RECIPE["adaptation_tokens"]
    stance_share = Fraction(adaptation["stance_and_event_tasks"], sum(adaptation.values()))
    check("Stance and event inputs as a share of adaptation tokens (%)", float(100 * stance_share), 1, "§4", tol=1e-9)
    for size, article in ((2, 1.99), (8, 7.7), (32, 27.5)):
        check(f"Cells of {size} sequences that contain a term carried by 1% of sequences (%)",
              float(100 * presence_probability(Fraction(1, 100), size)), article, "§4", tol=0.05)
    measured = contract_checks.run()
    part = measured["partition"]
    check("Sequences in the window", part["sequences"], 16, "§4")
    check("Language positions", part["denominators"]["language"], 756, "§4")
    check("Answer positions", part["denominators"]["answer"], 16, "§4")
    check("Stance positions", part["denominators"]["stance"], 2, "§4")
    check("Ranks", part["ranks"], 4, "§4")
    check("Microbatches per rank", part["microbatches_per_rank"], 2, "§4")
    check_true("Global denominators: gradient equal to the single-batch gradient up to single-precision rounding "
               f"(measured {part['global_relative_difference']:.1e})",
               0 < part["global_relative_difference"] < 1e-5, "§4, eq. 4.2", shown="about 1e-6")
    check("Per-cell averaging: relative difference from the single-batch gradient", part["local_relative_difference"], 0.86, "§4", tol=0.01)
    check("Cells that contain the stance term, of eight", part["cells_with_stance"], 1, "§4")
    check("Cells that contain the answer term, of eight", part["cells_with_answer"], 2, "§4")

    # ------------------------------------------------------------------ Section 5
    section("Execution that preserves the objective (Section 5)")
    proj = measured["projection"]
    check_true("Selected-position loss equals the all-position loss", proj["loss_difference"] < 1e-5, "§5, eq. 5.1")
    check_true(f"Their gradients agree up to single-precision rounding (measured {proj['gradient_relative_difference']:.1e})",
               0 <= proj["gradient_relative_difference"] < 1e-5, "§5, eq. 5.1", shown="about 1e-6")
    private, small = architecture.PRIVATE, architecture.TRIAL
    full = architecture.logits_bytes(2 * 2048, private.vocab)
    chosen = architecture.logits_bytes(2, private.vocab)
    check("2.4B microbatch, all positions: logits tensor (GiB)", full / 2 ** 30, 1, "§5", tol=1e-12)
    check("Its two supervised positions (MiB)", chosen / 2 ** 20, 0.5, "§5", tol=1e-12)
    check("Ratio", full / chosen, 2048, "§5")
    check("Trial batch of 4 × 512, all positions (MiB)", architecture.logits_bytes(4 * 512, small.vocab) / 2 ** 20, 64, "§5", tol=1e-12)
    check("Its four supervised positions (KiB)", architecture.logits_bytes(4, small.vocab) / 2 ** 10, 128, "§5", tol=1e-12)
    check("Output head's share of a position's matrix work at 2.4B (%)", 100 * architecture.projection_share(private), 7.0, "§5", tol=0.05)
    check("At 24M (%)", 100 * architecture.projection_share(small), 17.3, "§5", tol=0.05)
    check("Full-vocabulary loss minus label-restricted loss, two positions (nats)", proj["full_minus_restricted"], 15.66, "§5", tol=0.02)
    check("The same per position (nats)", proj["full_minus_restricted"] / 2, 7.8, "D6", tol=0.05)
    check("Log of the vocabulary size over three label tokens (nats)", log(small.vocab / 3), 7.9, "D6", tol=0.05)
    check("Ratio of a term's kept weight between cells of 32 and of 2 sequences",
          float(presence_probability(Fraction(1, 100), 32) / presence_probability(Fraction(1, 100), 2)), 14, "D4", tol=0.5)
    check_true("That difference is minus the log of the probability mass on the label tokens",
               abs(proj["full_minus_restricted"] - proj["minus_log_label_mass"]) < 1e-4, "§5")
    pack = measured["packing"]
    check("Examples packed", pack["segments"], 3, "§5")
    check("Packed length", pack["packed_length"], 64, "§5")
    check_true(f"Segmented attention: packed logits within 1e-5 of the unpacked ones (measured {pack['segmented_max_logit_difference']:.1e})",
               pack["segmented_max_logit_difference"] < 1e-5, "§5, eq. 5.2")
    check("Plain causal attention: largest logit difference", pack["plain_causal_max_logit_difference"], 2.9, "§5", tol=0.1)
    check("Typical logit magnitude", pack["typical_logit_magnitude"], 0.36, "§5", tol=0.01)
    check("Segmented attention: gradient into an earlier segment", pack["segmented_gradient_into_earlier_segment"], 0, "§5")
    check("Plain causal attention: gradient into an earlier segment", pack["plain_causal_gradient_into_earlier_segment"], 0.88, "§5", tol=0.02)
    check_true("The input embedding and the output head are one parameter object", measured["tied_head"], "§5")
    check("2.4B training state in single precision (GB)", architecture.training_state_bytes(private) / 1e9, 38.36, "§5", tol=0.005)

    # ------------------------------------------------------------------ Section 6
    section("The Trial (Section 6)")
    check("Trial parameters", trial["model"]["parameters"], 24255744, "§6")
    check_true("The formula gives the same count", architecture.parameters(small) == trial["model"]["parameters"], "§6, eq. 7.1")
    check("Tokenizer vocabulary", trial["tokenizer"]["vocab"], 8192, "§6")
    check("Task texts excluded from the tokenizer and the language corpora", trial["exclusion"]["task_texts_excluded_from_tokenizer_and_language_corpora"], 10686, "§6")
    speeds = [s["tokens_per_second"] for s in trial["language_training"]]
    check_true("Throughput between 3,200 and 3,500 tokens per second in every stage", all(3200 <= v <= 3500 for v in speeds), "§6")
    check("Peak allocated memory (GiB)", trial["hardware"]["peak_allocated_vram_bytes"] / 2 ** 30, 1.06, "§6", tol=0.005)
    check("Sequences per microbatch", trial["hardware"]["batch_size"], 4, "§6")
    check("Microbatches per update", trial["hardware"]["gradient_accumulation"], 8, "§6")
    seed, expansion, domain = trial["language_training"]
    check("Seed: updates", seed["updates"], 300, "§6")
    check("Seed: supervised tokens", seed["supervised_tokens"], 4914941, "§6")
    check("Seed: validation loss", seed["validation_loss"], 5.542, "§6", tol=0.0005)
    check("Expansion: updates", expansion["updates"], 1220, "§6")
    check("Expansion: supervised tokens", expansion["supervised_tokens"], 19988195, "§6")
    check("Expansion: validation loss", expansion["validation_loss"], 3.693, "§6", tol=0.0005)
    check("Expansion: loss on the seed's validation text before", expansion["seed_retention_loss_before"], 4.403, "§6", tol=0.0005)
    check("Expansion: after", expansion["seed_retention_loss_after"], 3.984, "§6", tol=0.0005)
    check("New general training tokens available to the expansion (millions)",
          trial["corpora"]["expansion"]["general_train"]["tokens"] / 1e6, 41.5, "evidence", tol=0.05)
    check("Adaptation: updates", domain["updates"], 74, "§6")
    check("Adaptation: supervised tokens", domain["supervised_tokens"], 1211960, "§6")
    check("Adaptation: validation loss on financial text", domain["validation_loss"], 3.238, "§6", tol=0.0005)
    check("Adaptation: loss on general text before", domain["general_retention_loss_before"], 3.984, "§6", tol=0.0005)
    check("Adaptation: after", domain["general_retention_loss_after"], 4.057, "§6", tol=0.0005)
    check("Cost of adaptation on general text (nats)", domain["general_retention_loss_after"] - domain["general_retention_loss_before"], 0.073, "§6", tol=0.0005)
    mix = trial["corpora"]["domain_mix"]
    check("Financial share of the adaptation mixture (%)", 100 * mix["financial_tokens"] / mix["tokens"], 80, "§6", tol=0.01)
    total = sum(s["supervised_tokens"] for s in trial["language_training"])
    check("Supervised tokens in all three stages", total, 26115096, "§6")
    check("Supervised tokens per parameter", total / trial["model"]["parameters"], 1.08, "§6", tol=0.005)
    review = trial["review"]
    check("Headlines reviewed", review["headlines_reviewed"], 2000, "§6")
    check("Decisions confirmed", review["decisions_confirmed"], 1993, "§6")
    check("Of which exclusions", review["excluded"], 114, "§6")
    check("Unresolved", review["unresolved"], 7, "§6")
    check("Three-class examples", review["three_class_candidates"], 1879, "§6")
    check_true("Confirmed decisions less exclusions give the three-class examples, and the three classes add up",
               review["decisions_confirmed"] - review["excluded"] == review["three_class_candidates"] == sum(review["by_class"].values())
               and review["decisions_confirmed"] + review["unresolved"] == review["headlines_reviewed"], "§6")
    check("Neutral", review["by_class"]["neutral"], 1219, "§6")
    check("Bullish", review["by_class"]["bullish"], 426, "§6")
    check("Bearish", review["by_class"]["bearish"], 234, "§6")
    check("Development, calibration and test examples each", review["split"]["development"], 101, "§6")
    check_true("The four partitions add up to the task pool", sum(review["split"].values()) == review["task_pool_with_a_literal_target"], "§6")
    check("Training: neutral", review["train_by_class"]["neutral"], 491, "§6")
    check("Training: bullish", review["train_by_class"]["bullish"], 140, "§6")
    check("Training: bearish", review["train_by_class"]["bearish"], 74, "§6")

    first, second = trial["sft"]["initial"], trial["sft"]["balanced"]
    support = first["development"]["support"]
    check("First model: epochs", first["epochs"], 2, "§6")
    check("First model: updates", first["optimizer_updates"], 45, "§6")
    check_true("It predicted neutral for every development and test example",
               first["development"]["predicted"]["neutral"] == 101 and first["test"]["predicted_neutral"] == 101, "§6")
    identity = accounting.majority_baseline(support)
    collapsed = accounting.summary(first["development"]["correct"], support, first["development"]["predicted"])
    check("Development: accuracy of the majority identity (%)", float(100 * identity["accuracy"]), 75.25, "§6, eq. 6.1", tol=0.005)
    check("Development: macro-F1 of the majority identity", float(identity["macro_f1"]), 0.2863, "§6, eq. 6.1", tol=0.00005)
    check_true("The first model's scores equal the identity exactly",
               collapsed["accuracy"] == identity["accuracy"] and collapsed["macro_f1"] == identity["macro_f1"], "§6")
    test_share = Fraction(first["test"]["correct"], first["test"]["examples"])
    check("Test: accuracy of the identity (%)", float(100 * test_share), 58.42, "§6, eq. 6.1", tol=0.005)
    check("Test: macro-F1 of the identity", float(2 * test_share / (1 + test_share) / 3), 0.2458, "§6, eq. 6.1", tol=0.00005)
    check("Cross-entropy of a predictor that outputs the training shares (nats)", accounting.prior_entropy(review["train_by_class"]), 0.81, "§6", tol=0.005)
    check("Rebalanced run: exposures per class per epoch", second["exposures_per_class_per_epoch"], 491, "§6")
    check("Epochs", second["epochs"], 3, "§6")
    check("Updates", second["optimizer_updates"], 139, "§6")
    check("Label positions", second["label_positions"], 4419, "§6")
    check_true("Label positions equal classes × exposures × epochs", second["label_positions"] == 3 * 491 * second["epochs"], "§6")
    balanced = accounting.summary(second["development"]["correct"], support, second["development"]["predicted"])
    article_rows = {"bearish": (10, 0.4545, 0.5000, 0.4762), "neutral": (76, 0.9394, 0.8158, 0.8732), "bullish": (15, 0.5000, 0.8000, 0.6154)}
    for label, (n, precision, recall, f1) in article_rows.items():
        check(f"{label.capitalize()}: support", support[label], n, "§6")
        check(f"{label.capitalize()}: precision", float(balanced["per_class"][label]["precision"]), precision, "§6", tol=0.00005)
        check(f"{label.capitalize()}: recall", float(balanced["per_class"][label]["recall"]), recall, "§6", tol=0.00005)
        check(f"{label.capitalize()}: F1", float(balanced["per_class"][label]["f1"]), f1, "§6", tol=0.00005)
    check("Rebalanced model: macro-F1", float(balanced["macro_f1"]), 0.6549, "§6", tol=0.00005)
    check("Rebalanced model: accuracy (%)", float(100 * balanced["accuracy"]), 78.22, "§6", tol=0.005)
    baseline = trial["sft"]["baseline"]["development"]
    check("TF-IDF baseline: macro-F1", baseline["macro_f1"], 0.6087, "§6")
    check("TF-IDF baseline: accuracy (%)", 100 * baseline["accuracy"], 75.25, "§6", tol=1e-9)
    trade = accounting.exchange(first["development"]["correct"], second["development"]["correct"])
    check("Directional examples gained", trade["directional_gained"], 17, "§6")
    check("Neutral examples given up", trade["neutral_lost"], 14, "§6")
    check("Correct after", sum(second["development"]["correct"].values()), 79, "§6")
    check("Correct before", sum(first["development"]["correct"].values()), 76, "§6")
    check("Gain in macro-F1", float(balanced["macro_f1"] - identity["macro_f1"]), 0.3687, "§6", tol=0.0001)
    check("Margin over the TF-IDF baseline", float(balanced["macro_f1"]) - baseline["macro_f1"], 0.0462, "§6", tol=0.0001)
    cal = trial["calibration"]
    check("Fitted temperature", cal["temperature"], 1.506, "§6", tol=0.0005)
    check("A raw confidence of 0.90 after temperature scaling", pm.temperature_scaled(0.90, cal["temperature"]), 0.81, "§6", tol=0.005)
    check("Review threshold", cal["review_threshold"], 0.574, "§6", tol=0.0005)
    check("Calibration examples for the temperature", review["calibration_split"]["temperature"], 51, "§6")
    check("For the threshold", review["calibration_split"]["threshold"], 50, "§6")

    # ------------------------------------------------------------------ Section 7
    section("Two scales, one contract (Section 7)")
    check("2.4B parameters", architecture.parameters(private), 2397459968, "§7, eq. 7.1")
    check("Parameters per layer at 2.4B", architecture.layer_parameters(private), 79631616, "§7")
    check("Parameters per layer at 24M", architecture.layer_parameters(small), 3343488, "§7")
    check("Twenty-eight layers", private.layers * architecture.layer_parameters(private), 2229685248, "§7")
    check("Shared vocabulary matrix", private.vocab * private.hidden, 167772160, "§7")
    check("Final norm", private.hidden, 2560, "§7")
    check("Head dimension at 2.4B", private.head_dim, 128, "§7")
    check("Head dimension at 24M", small.head_dim, 64, "§7")
    recipe = architecture.RECIPE
    check("Foundation tokens (billions)", recipe["foundation_tokens"] / 1e9, 50, "§7")
    check("Adaptation tokens (billions)", sum(recipe["adaptation_tokens"].values()) / 1e9, 5, "§7")
    check("Financial source text (billions)", recipe["adaptation_tokens"]["financial_source_text"] / 1e9, 4, "§7")
    check("General text (billions)", recipe["adaptation_tokens"]["general_text"] / 1e9, 0.75, "§7")
    check("Reading questions and answers (billions)", recipe["adaptation_tokens"]["reading_questions_and_answers"] / 1e9, 0.20, "§7")
    check("Stance and event tasks (billions)", recipe["adaptation_tokens"]["stance_and_event_tasks"] / 1e9, 0.05, "§7")
    check("Foundation tokens per parameter", architecture.tokens_per_parameter(recipe["foundation_tokens"], private), 20.9, "§7", tol=0.05)
    check("Human-labelled records", sum(recipe["labelled_records"].values()), 32000, "§7")
    check("Scale factor between the two configurations", architecture.parameters(private) / architecture.parameters(small), 100, "§7", tol=1.5)

    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, int) or float(x).is_integer():
            return f"{int(x):,}"
        return f"{x:.4f}"

    def art(x):
        if isinstance(x, str):
            return x
        if isinstance(x, int) or float(x).is_integer():
            return f"{int(x):,}"
        return f"{x:g}"

    count = sum(len(r) for _, r in sections)
    agree = sum(ok for _, r in sections for *_, ok in r)
    lines = ["# Results", "", "Every figure in the article, recomputed.", "", f"{agree} of {count} figures agree with the article", ""]
    for title, items in sections:
        lines += [f"## {title}", "", "| Quantity | Recomputed | Article | Where | Agrees |", "| --- | ---: | ---: | --- | :---: |"]
        for label, value, article, where, ok in items:
            lines.append(f"| {label.replace('|', chr(92) + '|')} | {fmt(value)} | {art(article)} | {where} | {'✓' if ok else '✗'} |")
        lines.append("")
    (ROOT / "reproduce" / "results.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    print(f"{agree} of {count} figures agree with the article")
    for title, items in sections:
        for label, value, article, where, ok in items:
            if not ok:
                print("  DISAGREES:", title, "|", label, "|", value, "vs", article)
    return 0 if agree == count else 1


if __name__ == "__main__":
    raise SystemExit(main())
