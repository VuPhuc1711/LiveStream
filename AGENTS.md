# Project data review and reporting policy

- Codex reviews new data autonomously against the agreed business label definitions; do not wait for the user to approve each sentence.
- Record AI review as `review_status = DA_DUYET_AI`, `reviewer = Codex AI`. Do not represent AI review as human approval.
- A subsequent user label correction takes precedence: record `review_status = DA_DUYET`, `reviewer = Vũ Hồng Phúc`, and preserve the prior label and the reason for the change.
- Review text, reason, cue, and category together. Resolve ambiguity by correcting the label or rewriting the sentence before evaluation, preserving provenance. Never force labels to maintain class balance.
- KHONG_CANH_BAO: ordinary descriptions, visible features, operations, or safe-use guidance without an objective claim requiring evidence or deception.
- CAN_XAC_MINH: objective claims requiring specifications, documents, certification, origin, price, conditions, or evidence, without clear intentional falsehood.
- CANH_BAO: admitted falsehood, concealment, alteration or distortion; dangerous or absolute promises; deliberate misleading statements or dangerous guidance.
- Never use model predictions to establish ground truth. Never include tests or their transcripts in training/validation. Freeze test labels and evaluation criteria before inference; do not revise a locked test in response to predictions.
- Check identifiers, required fields, encoding, exact/normalized duplicates, family separation, and semantic overlap. Report briefly only if a serious issue cannot be resolved safely.
- Run training, validation, error analysis, and stability checks when they directly affect model quality and are within the requested scope. Do not interpret this policy as permission to train or evaluate when the user requested preparation only.
- Keep reporting short: results, material errors, technical decision, next step. No long Markdown reports, full lists of correct predictions, or duplicate artifacts. Retain artifacts needed for the model, reproducibility, and verification.
- Keep Rules and Hybrid thresholds unchanged unless the user explicitly authorizes changes. Change the pipeline checkpoint only when its predeclared promotion criteria pass.
