# Requirements Clarification Questions

Please answer each question by filling the letter after the `[Answer]:` tag.
Let me know when you're done.

---

## Question 1
Should the security extension rules (SECURITY-01 to SECURITY-15) be enforced
as hard constraints during this workflow?
(This is a portfolio/PoC ML project — skipping is appropriate here.)

A) Yes — enforce all SECURITY rules (recommended for production applications)
B) No — skip security rules (suitable for PoCs and portfolio projects)
X) Other (please describe after [Answer]: tag below)

[Answer]:No

---

## Question 2
Which model type should be the primary implementation target?

A) Custom CNN only (GeometricCNN)
B) Pretrained ResNet-18 with frozen backbone only
C) Both — implement both and let the config switch between them
X) Other (please describe after [Answer]: tag below)

[Answer]:B) Custom

---

## Question 3
What should the evaluate.py output?

A) Console report only (sklearn classification_report + accuracy)
B) Console report + confusion matrix plot saved to disk
C) Console report + confusion matrix + per-class sample visualisation
X) Other (please describe after [Answer]: tag below)

[Answer]: C)
