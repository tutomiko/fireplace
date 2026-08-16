# Feature Island Induction From A Semantically Rich Ocean

Whether beneath lowering tides or amidst embers slowly burning away, what remains are only the regions grounded enough to endure, distilled into their concrete form through the surrounding semantic structure.

This repository implements that idea as an open world heatmap generator, approached through the lens of thermodynamics rather than a fixed classifier head. The implementation began as an observation about watching embers cool in a fireplace, and the code kept the metaphor: heat, decay, dissipation, and what is left behind once the fire settles. The heatmap however resembled something more like an ocean, so that's where the more grandiose name is derived from.

Read the code before taking any of this on faith. The behavior described below is not aspirational, it is what `src/fireplace` actually does.

![Demo](demo/demo.gif)

## What problem this solves

Traditional object detectors are trained against a fixed, closed set of classes. If a class was not in the training data, the model has no way to detect it. This project takes a different approach.

In the demo harness, the user defines a class at request time, not at training time, by lassoing region(s) of interest on a reference image. The system embeds both the reference image [left] and the target image with a self-supervised vision transformer (DINOv2-b), then uses the embedding similarity between the lassoed region and every patch of the target image [right] to build a heat distribution over the target. No fixed taxonomy, no retraining, no closed label set. The class is whatever the user points at.

## The thermodynamics framing

The algorithm is organized around three stages, each named for a physical process rather than a machine learning term, because that is a fair description of what the math is doing.

**Induction.** A heat map is formed by comparing every patch embedding in the target image against one or more class centroids derived from the user's lasso selections. Regions that are semantically close to the centroid run hot. Regions that are not run cold. This is `form_centroid_heatmap` in `inductor.py`.

**Enrichment.** The wide, diffuse heat from induction is narrowed using patch-level cosine similarity against the individual embeddings inside the lasso, not just the centroid. Non-matching regions drop off sharply below a similarity threshold. This is `enrich_heat`.

**Dissipation.** The heat map is progressively cooled. A floor cutoff rises over a series of steps, and cooler regions are extinguished as that floor climbs, while surviving hotspots also decay in intensity. This is `dissipate_heat`, and it runs twice per induction cycle, on two different inputs and for two different purposes: the first pass runs on the raw centroid heat, before enrichment, to cull the bulk of false positives early; the second pass runs after enrichment, on its sharper, patch-filtered heat, to separate distinct instances and reduce semantic bleed between neighboring regions.

What survives the full cycle of induction, enrichment, and dissipation is not the whole heat field. It is only the regions whose signal was strong and coherent enough to remain once the process settles, in the same way that only the driftwood grounded firmly enough survives an outgoing tide, or only the embers with enough remaining fuel are still glowing once a fire dies down.

## Output

The output of the full induction cycle is a single continuous heat map over the target image: a per-patch intensity grid, normalized to [0, 1], with everything below the settled floor already extinguished to zero. There is no downstream segmentation or bounding-box step — what you get is the heat field itself, not masks or boxes derived from it.

## Repository layout

```
src/fireplace/
  inductor.py   Fireplace class: induction, enrichment, dissipation
  heatmap.py    FireplaceHeatmap: the settled heat mask and its bounds
  __init__.py   Public exports

demo/
  harness/      A FastAPI + DINOv2 demo harness with a browser UI
                for lassoing a reference image and streaming the
                resulting heat map onto a second image in real time

tests/
  test_inductor.py   Unit tests against Fireplace / FireplaceHeatmap
```

## Running the demo harness

The harness in `demo/harness` is the fastest way to see the algorithm work end to end. It serves two images side by side. Lasso a region on the left image to define a class, and the right image streams back a live heat map as it settles.

```
pip install -r requirements.txt   # fastapi, torch, transformers, scipy, pillow, matplotlib
python demo/harness/server.py
```

The first request downloads DINOv2 base weights, so expect a pause on first run.

## Tuning parameters

Two parameters govern how much of the heat field survives.

- `decay`: how quickly surviving heat loses intensity during dissipation. Higher values cool the scene faster. Defaults to `0.01`.
- `decay_threshold`: the similarity and heat floor used during both enrichment and dissipation. Lower values are more permissive and leave more of the image lit. Higher values are stricter and leave less. Defaults to `0.2`.

Both are exposed as sliders in the test harness, and both are passed straight through to `Fireplace.set_decay` and `Fireplace.set_decay_threshold`.
