# Amazon ML Challenge 2026: Business Entity Resolution

**Final result: rank 263 of 10,500+ teams.** Best public leaderboard score **0.987859** (macro F0.5).

This repository holds the code of our solution: candidate generation, pair scoring, the decision rule, and the
test-time corrections that gave most of the late leaderboard gains. Competition data, predictions and trained models
are not included.

## The task

Three sources of business records (name + address). Source 1 is the reference list of entities; each record in
Source 2 and Source 3 belongs to at most one Source 1 entity, or to none. For every Source 1 entity, predict the set of
Source 2/3 records that describe the same business.

- Training data: India and US. Test: India, US and **France (a country never seen in training)**.
- Metric: F0.5 computed per Source 1 entity, then averaged. Precision counts double, and an entity correctly
  predicted as empty scores 1.0.
- Leaderboard weight: about 85% India + US, 15% France.

## Leaderboard progression

| Submission | Main change | Public LB |
|---|---|---|
| v1 | blocking + rapidfuzz features + LightGBM | 0.948 |
| v3 | name-skeleton / street blocking, XGBoost | 0.965 |
| v10 | wider blocking, e5 cross-encoders, listwise context | 0.979 |
| v12 | France decisions from a model that transfers to unseen countries | 0.980171 |
| v13 | e5-base dense retrieval add-on + joint re-score | 0.98255 |
| v19 | joint model v9 (27 inputs), France blend with e5-large / Qwen scores | 0.985183 |
| A' | remove US near-number decoys ("odd one out") | 0.986279 |
| A2 | more US decoy cells (mixed siblings, street differs) | 0.987131 |
| A8 | density-ratio decoy estimator, India restores, France adds, source cap | 0.987711 |
| **A14** | France decoys + name-only model trained under the test condition | **0.987859** |

## Pipeline

```
S1 / S2 / S3 records
   |
1. Normalise: canonical name, address key, house numbers, legal form, learned native-script -> Latin word dictionary
   |
2. Candidates: key-based cascade  +  dense retrieval (e5 bi-encoder, forward top-k and reverse top-1)   ~5 per entity
   |
3. Scoring (India/US): stage-1 XGBoost -> stage-2 context XGBoost (+ small e5 cross-encoders)
                       -> joint XGBoost over 27 inputs (stage probabilities, dense cosine,
                          e5-small / listwise / e5-large / bge-m3 cross-encoder scores, competition features)
   |
4. Decision: each record to at most one entity, then per entity the prefix of candidates that maximises expected F0.5
   |                                        France: separate recipe (models that transfer to unseen countries,
   |                                        Qwen2.5-7B score mix, own-address rules, near-number veto)
   |
5. Test-shift corrections (post-processing, learned from train vs test differences)
   |
matching_results.tsv + candidate_pairs.tsv
```

**Validation.** Two entity folds on a "harder split" that drops 18.6% of Source 1 entities (and keeps their records),
because test has about 1.1 more records per entity than train. Final India/US validation F0.5: India 0.9918, US 0.9904.

## What moved the score late in the competition

Most gains in the last two days came from **differences between train and test**, not from bigger models.

1. **Injected decoys in US test.** Test US contains extra records with the same name, the same street and a house
   number changed by a small amount, whose other matched records all agree on the entity's number ("odd one out").
   Train has far fewer. Comparing train TRUE matches per entity with test ACCEPTED pairs per entity, cell by cell,
   shows where the excess is; removing those pairs gained +0.0011, +0.00085 and +0.00058 on three submissions.
   `src/nearby_number.py`, `src/shift3.py`, `src/shift_all.py`.
2. **Density-ratio decoy estimator.** A classifier separating train accepted-TRUE pairs from test accepted pairs
   (text, number-gap and sibling-agreement features only) gives, for each test pair, an estimated false fraction
   f = 1 - 1/r. It was checked on a null split (no shift), calibrated against the three leaderboard gains, and used
   to add removals and restore pairs earlier rules removed wrongly. `src/dratio.py`, `gpu/gpu9/dr_ens.py`.
3. **India has no such decoys.** The same comparison on India shows no excess, so India removals were reverted.
4. **France is flooded with the same decoy family** (0.60 near-number candidates per entity vs 0.09 in India).
   France has no labels, so its accepted rate was compared with its own clean near-number pairs, using the ratio
   that holds in both US and India truth. `src/fr_decoy.py`.
5. **Test has no orphan name-only records.** The harder split leaves 20% of address-less records without an owner;
   test has exactly the full-train density, so the main model is too cautious on them. A separate name-only model,
   trained on validation pairs with the orphans removed, adds high-confidence name-only matches
   (+0.0002 to +0.0003 F0.5 on validation). `src/nameonly_model.py`.
6. **Generator hard cap.** Every train entity has at most 5 Source-2 and 6 Source-3 records; test assignments
   above the cap were trimmed.

## What did not help

- Bigger or different models on the same inputs: LightGBM / CatBoost swaps, a set transformer, Qwen2.5-7B scores as a
  feature, per-country models, test-reweighted retraining, LLM group judgment (all within +/- 0.0001 on validation).
- Pseudo-labelling on test (it relearned our own decisions and undid proven decoy removals).
- Rules for ambiguous name-only records (same name, several entities): the model is exactly calibrated there,
  the owner cannot be identified from the data.
- Fuzzy owner search for name-only records without candidates (under 8% precise).

## Where the remaining loss is (estimate)

| Bucket | Share of loss |
|---|---|
| Name-only records shared by several same-name entities | about 45% |
| France (unseen country, no labels) | about 33% |
| India / US addressed pairs | about 17% |

## Repository layout

| Path | Contents |
|---|---|
| `src/` | normalisation, blocking, features, training, decision rule (`decide.py`), evaluation, France recipe, all test-shift analyses and submission builders |
| `gpu/gpu2` .. `gpu/gpu5` | A100 packages: dense retrieval, cross-encoders (e5, bge-m3), Qwen2.5 LoRA scoring, LLM group judgment |
| `gpu/gpu8` | test-reweighted retraining experiment |
| `gpu/gpu9` | bagged density-ratio estimator (XGBoost + MLP) |
| `docs/DISCLOSURE_France.md` | disclosure of hand-inspected France pairs (allowed by the organisers when disclosed) |

## Notes

- Scripts use absolute paths from the original machine (`C:/ML/...`, `D:/amazon_er/...`); adjust before running.
- The competition data is not redistributed here.
- Models used: XGBoost, LightGBM, multilingual-e5, bge-m3, Qwen2.5 (up to 7B), all under MIT or Apache licences.
- Memory-heavy jobs were run through `src/guard.sh`, which stops a job before the machine runs out of memory.
