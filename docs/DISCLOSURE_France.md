# Disclosure: constructed / hand-inspected test labels (France), night of 2026-09-27

Per the organisers' updated rules (as relayed by the team, email in the last 24 h), constructed test labels are allowed
when disclosed. This is everything we did with test (France) data beyond running the trained pipeline:

1. Hand audit (reading, no labels written into any model): 40 random France entities with all candidates
   (audit/audit_0.txt), and 25-pair random samples of every France change v13 -> v14 (audit/delta_*.txt: pairs the
   blend removes / adds, pairs each rule adds). Used ONLY to judge whether a change looks right (precision estimate) and
   to find error patterns. Outcome: rules R2 (pseudo-word / single-prefix truncation / acronym / extra word / truncation
   at the entity's own address) judged 24-25/25 correct per sample; blend removals ~50% true (net positive under F0.5);
   blend additions ~60% true, errors = near-number sister companies -> VETO rule.
2. Rules are generic generator rules; each was first validated on the unseen-country proxy (India / US scored by a model
   that never saw that country, TRUE labels) before touching France: precision 0.93-0.99, F0.5 up in both countries
   (VETO: up in US, down in India; applied to France because France's shifted-number decoys match the US pattern and
   the hand audit).
3. No France label was used to train any model. Self-training on France was tested on the proxy, failed in one country,
   and was NOT used.

Addendum (07:10): a further 36 France pairs (18 removed / 18 added by the e5-large mix) were read and judged; they
led to the PROTECT rule, which was then validated on the proxy (India / US true labels) before use.
