# Feature Island Induction From A Semantically Rich Ocean

Whether beneath lowering tides or amidst embers slowly burning away, what remains are only the regions grounded enough to ensure, distilled into their concrete form through the surrounding semantic structure.

This repository implements that idea as an open world class detection algorithm, approached through the lens of thermodynamics rather than a fixed classifier head. The implementation began as an observation about watching embers cool in a fireplace, and the code kept the metaphor: heat, decay, dissipation, and what is left behind once the fire settles. The heatmap however resembled something more like an ocean, so that's where the more grandiose name is derived from.

Read the code before taking any of this on faith. The behavior described below is not aspirational, it is what `src/fireplace` actually does.

## What problem this solves

Traditional object detectors are trained against a fixed, closed set of classes. If a class was not in the training data, the model has no way to detect it. This project takes a different approach.

In the public test harness, the user defines a class at request time, not at training time, by lassoing region(s) of interest on a reference image. The system embeds both the reference image [left] and the target image with a self-supervised vision transformer (DINOv2-b), then uses the embedding similarity between the lassoed region and every patch of the target image [right] to build a heat distribution over the target. No fixed taxonomy, no retraining, no closed label set. The class is whatever the user points at.

## The thermodynamics framing

The algorithm is organized around three stages, each named for a physical process rather than a machine learning term, because that is a fair description of what the math is doing.

**Induction.** A heat map is formed by comparing every patch embedding in the target image against one or more class centroids derived from the user's lasso selections. Regions that are semantically close to the centroid run hot. Regions that are not run cold. This is `form_centroid_heatmap` in `inductor.py`.

**Enrichment.** The wide, diffuse heat from induction is narrowed using patch-level cosine similarity against the individual embeddings inside the lasso, not just the centroid. Non-matching regions drop off sharply below a similarity threshold. This is `enrich_heat`.

**Dissipation.** The heat map is progressively cooled. A floor cutoff rises over a series of steps, and cooler regions are extinguished as that floor climbs, while surviving hotspots also decay in intensity. This is `dissipate_heat`, and it runs both before and after enrichment, since the input heat distribution is different each time.

What survives the full cycle of induction, enrichment, and dissipation is not the whole heat field. It is only the regions whose signal was strong and coherent enough to remain once the process settles, in the same way that only the driftwood grounded firmly enough survives an outgoing tide, or only the embers with enough remaining fuel are still glowing once a fire dies down.

## From heat to detections

A cooled heat map on its own is not a detection. The final step, implemented in `heatmap.py`, treats every remaining region of nonzero heat as a connected component, using 8 connected connected component labeling over the settled heat grid. Each connected component is a thermal island.

Every thermal island produces two artifacts:

- A binary mask, exactly the shape of the surviving region, for pixel accurate localization.
- A bounding box, derived from the extent of that mask, for coarse localization and downstream consumption by anything that expects boxes instead of masks.

This is the open world equivalent of instance segmentation and bounding box detection, except the class was defined by a lasso a moment ago instead of by a label in a training set.

## Repository layout

```
src/fireplace/
  inductor.py   Fireplace class: induction, enrichment, dissipation
  heatmap.py    FireplaceHeatmap and ThermalIsland: masks, bounding boxes
  __init__.py   Public exports

tests/
  harness/      A FastAPI + DINOv2 test harness with a browser UI
                for lassoing a reference image and streaming the
                resulting heat map and thermal islands onto a
                second image in real time
```

## Running the test harness

The harness in `tests/harness` is the fastest way to see the algorithm work end to end. It serves two images side by side. Lasso a region on the left image to define a class, and the right image streams back a live heat map, settling into purple masks and green bounding boxes around the surviving thermal islands once the process converges.

```
pip install -r requirements.txt   # fastapi, torch, transformers, scipy, pillow, matplotlib
python tests/harness/server.py
```

The first request downloads DINOv2 base weights, so expect a pause on first run.

## Tuning parameters

Two parameters govern how much of the heat field survives to become a detection.

- `decay`: how quickly surviving heat loses intensity during dissipation. Higher values cool the scene faster.
- `decay_threshold`: the similarity and heat floor used during both enrichment and dissipation. Lower values are more permissive and leave more of the image lit. Higher values are stricter and leave less.

Both are exposed as sliders in the test harness, and both are passed straight through to `Fireplace.set_decay` and `Fireplace.set_decay_threshold`.
