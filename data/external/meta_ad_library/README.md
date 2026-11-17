# Sephora ads from the Meta Ad Library: audience test for adsim

Downloaded 14 Nov 2026 through Apify (`curious_coder/facebook-ads-library-scraper`) from three
public Ad Library pages. Engine code tested: branch `data/kaggle-benchmark`.

## Read this first

- **The Ad Library publishes no clicks, spend, impressions, CTR or CPM for commercial ads.**
  Nothing here measures the engine's accuracy on Sephora's actual results.
- Everything in `real/` is copied from the public record. Everything in `simulated/` is generated.
  The two are never mixed in one file without a label.
- The US Sephora page has no audience or reach data (Meta discloses that for EU delivery only).
  Its 500 ads are in `real/ads.csv` but take part in no test.

## What was downloaded (real)

| Page | Ads | Distinct creatives | Launch dates | Audience and reach data |
|---|---:|---:|---|---|
| Sephora France | 1,003 | 360 | 10 Aug to 13 Nov 2026 | yes, 17 targeting set-ups |
| Sephora Italy | 1,000 | 445 | 16 Jun to 14 Nov 2026 | yes, 21 targeting set-ups |
| Sephora (US page) | 500 | 177 | 2 to 14 Nov 2026 | none |

These are the most recent ads per page up to the download limit, not every ad the page ever ran.
For 1,972 of the 2,003 EU ads the reach by age and gender adds up to within 5% of Meta's total.

| File | One row per | Contents |
|---|---|---|
| `real/ads.csv` | library ad | page, start and end date, format, call to action, platforms, targeting (location, gender, age range), total EU reach, copy |
| `real/creatives.csv` | creative | the same copy in the same format on the same page |
| `real/reach_by_segment.csv` | ad x country x age band x gender | people reached |
| `real/real_manifest.json` | | source runs, checksums, counts, caveats |

## What was simulated, and how

`scripts/meta_ad_library/simulate.py` adds daily impressions, clicks and spend, because the engine
cannot run without them. The recipe was fixed before the engine was run on the result; every
parameter and every drawn effect is in `simulated/simulation_spec.json` (seed 20261114).

- **Kept real:** which ads ran on which days, their copy and format, their launch date, the
  age x gender audiences they reached and how many people in each.
- **Audience:** country x gender x age band (12 per country). Ad x audience cells with fewer than
  1,000 people reached are dropped.
- **Impressions:** real reach x 1.5, spread evenly over the ad's real live days, with day noise.
- **Clicks:** a hidden click rate per ad, audience and day, built from an audience effect, a format
  effect, a lift for promotional wording, a creative effect, real delivery skew, fatigue, weekday
  and noise. Clicks are drawn from it.
- **Spend:** a hidden price per audience and format with a slow upward trend.
- **Window:** 1 Sep to 13 Nov 2026. Train = first 52 days. **Test = last 22 days, 48,809 rows,
  35% of all rows** (larger than the usual 15% so there is more to review).

`simulated/truth.csv.gz` holds the hidden click rate and price behind every row, with each
component, so any forecast can be checked against what was planted. The engine never reads it.

## Results

Full numbers: `results/benchmark_results.json`. Lower error is better.

### Part A: real data only

**A1. Does the copy predict how long an ad runs or how many people it reaches?** No.
Trained on 1,027 earlier finished ads, scored on 448 later ones.

| Outcome | Copy model (engine's external-evidence path) | Format average |
|---|---:|---:|
| Days running, rank correlation | -0.14 | 0.00 |
| EU reach, rank correlation | 0.26 | 0.39 |
| EU reach, copy never seen before | 0.09 | 0.37 |

**A2. Does the creative tell us which audience an ad reaches?** Almost nothing beyond the targeting.
Trained on 1,153 earlier ads, scored on 530 later ones. "Share misallocated" is the fraction of
reach placed in the wrong age x gender group (0 = perfect).

| Forecast of the reach split | Share misallocated | Largest audience named correctly |
|---|---:|---:|
| Page average | 0.330 | 53% |
| Average of ads with the same targeting | 0.243 | 57% |
| Same targeting + creative features | 0.240 | 57% |

### Part B: full engine on simulated performance (48,806 test rows scored)

| Forecast | CTR error vs planted truth | Click error share | CPM error vs observed |
|---|---:|---:|---:|
| **adsim engine** | **0.148** | **0.231** | 0.079 |
| Last 7 days | 0.243 | 0.259 | **0.075** |
| Audience average | 0.341 | 0.269 | 0.099 |
| Advertiser average | 0.463 | 0.390 | 0.222 |

| Audience question | adsim engine | Audience average | Chance |
|---|---:|---:|---:|
| Orders the 12 audiences correctly, France (rank correlation) | 0.93 | 0.90 | |
| Orders the 12 audiences correctly, Italy | 1.00 | 1.00 | |
| Names the best audience for a new creative (165 creatives) | 74.5% | 72.7% | 19% |

- CTR error on new creatives: engine 0.198, audience average 0.244, last 7 days 0.228.
- The engine under-predicts total test clicks by 8.7% and CPM by 4.9%; the CPM gap grows from
  2.7% in test week 1 to 6.6% in week 3, because the planted price trend has no seasonal index
  to explain it.
- 80% ranges contained the outcome 80% (seen ads) and 79% (new ads) of the time for CTR, and
  80% / 78% for CPM.
- No forecast exceeded 100% CTR.

## What this does and does not show

- The engine ran unchanged on a new advertiser, in French and Italian, with 575 creatives,
  8,637 ad x audience pairs and 15% zero-click rows.
- On simulated performance it recovers audience levels and beats simple baselines on CTR. The
  simulated click rate is built from additive effects, which is the shape the engine assumes,
  so this is a favourable test.
- Picking the best audience for a new creative, it is no better than "use the audience that
  usually performs best". It does not match creatives to audiences.
- On real data, copy does not predict longevity or reach, and adds nothing to targeting for
  predicting who an ad reaches. The engine's `external_evidence` (ad-library) feature is not
  supported by this data.

## Files for review

| File | Use |
|---|---|
| `results/test_predictions.csv.gz` | every scored test row: simulated outcome, planted truth, engine forecast with ranges, three baselines, new-ad and new-creative flags |
| `results/test_review_sample.csv` | 600 readable rows (100 per page x seen ad / new ad / new creative) with headline and copy |
| `results/audience_ranking.csv` | per audience: observed, planted and forecast CTR |
| `results/new_creative_audience_picks.csv` | per new creative: true best audience and each method's pick |
| `results/audience_archetypes.csv` | engine archetypes next to the planted audience effects |
| `results/real_longevity_predictions.csv`, `results/real_audience_share_predictions.csv` | Part A, row by row |

## Reproduce

```bash
python scripts/meta_ad_library/fetch.py --limit 1000 --tag full_eu --pages sephora_france sephora_italia   # needs APIFY_TOKEN; costs about $1.45
python scripts/meta_ad_library/fetch.py --limit 500  --tag full_us --pages sephora                          # about $0.35
python scripts/meta_ad_library/build_dataset.py full_eu full_us
python scripts/meta_ad_library/simulate.py
python scripts/meta_ad_library/benchmark.py        # needs fastembed (requirements-optional.txt)
```

A new download returns whatever is live that day, so `real/` will differ; `simulate.py` and
`benchmark.py` are deterministic given `real/`. Raw responses are kept locally in `raw/`
(git-ignored).
