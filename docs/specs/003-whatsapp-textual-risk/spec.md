# Feature specification: WhatsApp textual-risk classifier

## Objective

Provide an auxiliary score answering: **does this WhatsApp message resemble textual
patterns associated with misinformation in the FakeWhatsApp.BR corpus?** The score is
not a truth probability and cannot override the evidence-first verdict.

## Inputs and routing

- WhatsApp message text, regardless of length: WhatsApp-risk model.
- Article extracted from a URL: existing article/editorial-risk model.
- Message plus URL: compute both scores independently.
- Media-only or media-dependent message: abstain and request media analysis.

## Data contract

- Use only existing `misinformation` labels `0` and `1`.
- `-1` means unlabeled and must never be converted to a negative example.
- Do not use user id, telephone location, group id, or other personal/social attributes.
- Collapse exact normalized duplicates and exclude groups spanning temporal splits.
- Train through August 2018, validate on September, test on October.

## Outputs

`probability_textual_risk`, an operational band, model/version metadata, and explicit
limitations. Public APIs must not name this value `probability_fake`.

## Acceptance gates

- Test ROC-AUC >= 0.80.
- Test macro-F1 >= 0.75.
- ECE <= 0.10 after calibration.
- Transformer must improve macro-F1 by at least 0.02 over the best transparent baseline
  or provide a documented operational benefit.
- All reported evaluations must be duplicate-group and temporally isolated.
