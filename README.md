# dtg-knockout

**A small, fast, specialized matting model for print-on-demand (DTG/DTF) graphics.**

Given a design rendered on a *known* solid background color (a garment color), `dtg-knockout`
produces a clean, print-ready binary cutout — keeping every part of the artwork and removing
only the background. One ~2M-parameter forward pass, ~100 ms on CPU, deterministic, no API.

It is deliberately **not** a general background remover. It solves one narrow problem extremely
well by exploiting a fact general tools throw away: **we already know the background color.**

---

## The problem

Print-on-demand / direct-to-garment (DTG) pipelines generate artwork on a solid garment-color
field, then must "knock out" that field to a transparent, print-ready graphic. Crucially, the
output must be **binary alpha** — every pixel is either fully printed or fully absent, with **no
semi-transparency** — because a printhead lays down ink or it doesn't. On dark garments this
matters twice over: a **white underbase** layer is printed first (so colors stay vivid), and that
underbase is generated *from the alpha mask*. A feathered, semi-transparent edge has no clean
meaning at the printhead — it produces muddy edges, halos, or misregistration between the white
underbase and the color layers. The knockout has to:

1. Remove the background, including enclosed regions (letter loops, gaps between elements).
2. **Keep everything else** — including disjoint elements: a caption, a separate logo, a tagline
   set apart from the main graphic.
3. Produce clean edges with no garment-color halo.
4. Not shred artwork whose colors are close to the garment color.

## Why general background removers aren't the right fit

The obvious approach is an off-the-shelf background remover. Tools like **rembg**, **Photoroom**,
and **BiRefNet** are good at what they're built for — general subject cutouts for photos,
e-commerce, and design. DTG knockout is a *different, narrower* problem, and that's where the fit
breaks down:

- **They output semi-transparent, anti-aliased alpha.** Every major background remover
  (rembg, Photoroom, BiRefNet, generative matting) feathers its edges with soft, partial alpha —
  ideal for compositing onto a photo, *wrong for print*. DTG and white-underbase printing need a
  hard binary mask; a soft matte forces a downstream binarization that its feathered edges survive
  badly (halos, jaggies, or an over-aggressive choke that eats the artwork). This model produces
  binary alpha directly.
- **They do salient-object segmentation** — "find the one main subject" — trained on natural-image
  datasets (DUTS, COCO). rembg is a 44M-parameter U²-Net; Photoroom is a paid segmentation API.
  Both answer *"what is the subject?"*, not *"what isn't the background?"*
- **So they delete disjoint elements.** A tagline or logo set apart from the main graphic isn't
  "salient," so it gets dropped. For a print design that is a broken product.
- **They ignore the one thing we know for certain — the background color.** They are
  general-purpose and get no hint; they guess.
- **They don't use the one signal we always have — the garment color.** Being general-purpose,
  they infer the background (or ignore color entirely) rather than being told exactly which color
  to remove — the piece of information a print pipeline always knows.
- **And the practical costs add up.** Commercial APIs can get expensive at volume; generative /
  diffusion-based removers are non-deterministic and can *alter the artwork itself* — a redrawn
  edge or a hallucinated detail is unacceptable on a design that must print exactly as authored;
  and some are slow. A small, conditioned, deterministic model sidesteps all three.

## Why not just key the color?

A color keyer decides each pixel from one number: its distance from the background color. That
only works when artwork and background never share a color, and garment graphics share it
routinely — inline strokes and distress drawn in the garment color are, pixel for pixel, identical
to the field around them. A rendered field also drifts off the requested color, so a tolerance
wide enough to clear it eats the artwork's near-ground tones. And an anti-aliased edge is a blend
of ink and background that a keyer producing a hard mask can only keep, as a fringe, or drop, as a
jagged edge.

No threshold resolves these. The answer lives in the neighborhood — whether ink encloses the
pixel, continues a stroke, sits on a contour — and a per-pixel rule cannot see it. A model that
looks at the neighborhood can. It takes the keyer's own distance map as an input and learns only
what the keyer gets wrong: the decisions that need context, not color.

## The key insight: we know the background color

Reframe the task. It is not *"segment the subject."* It is:

> **"Remove the known background color. Keep everything else."**

That single reframe changes everything:

- It **keeps disjoint elements by construction** — we are not hunting for a subject, so a separate
  caption or logo is kept like any other non-background pixel. This is precisely the case rembg
  deletes.
- It lets us **condition the model on the known background color** — a signal no general matting
  model has.

Concretely, the model input is **4 channels**: `RGB` + a **color-distance-to-known-background map**.
That distance channel is the exact per-pixel signal a deterministic color keyer computes — how far
each pixel is from the *shaded* background segment `{k·bg}` (background under lighting is
approximately a scalar multiple of the base color). So the model **starts from the color-keyer's
strength** and only has to learn the *residual* the keyer gets wrong:

- **tonal collision** — artwork whose color is close to the garment color,
- **soft / anti-aliased edges** and garment-color halos,
- **thin structures** (fine strokes, small text).

Because the hard semantic work (what is background) is largely handed to the model as an input
channel, a **~2M-parameter U-Net is enough** — vs a 44M-parameter general segmenter — and it runs
in ~100 ms with no API and no retries.

## How it's trained

**Free labels via synthetic composition.** Any transparent PNG's own alpha channel *is* a
ground-truth matte. To make a training pair we composite that artwork onto a garment color with
production-matched artifacts and record the *requested* garment hex:

- global tint drift (the rendered ground drifts off the requested hex, as real generators do),
- low-frequency shading, soft drop shadows,
- anti-aliasing (supersample → downscale), sensor/print noise, JPEG round-trip,
- occasional multi-element layouts (a graphic + a separate caption) to teach *keeping* disjoint
  elements.

The recorded `bg_hex` is the color the pipeline *knows*; the pixels drift *off* it — mirroring
production, where you know the requested color and the render wanders. This yields unlimited,
perfectly-labeled `(image, bg_hex) → alpha` pairs at zero labeling cost.

**Diversity beats volume.** The foreground pool spans multiple graphic *styles* (POD designs,
clipart, stickers). The empirical finding driving the design: adding a new style lifted **every**
held-out style at once — including the in-distribution one — with no trade-off. The known-bg
conditioning generalizes cleanly; the gap to "any graphic" is *style diversity*, not model
capacity, and not deeper background wiring.

**Binary alpha for DTG.** Targets are binarized (0/255) and the model's output is thresholded to
binary — DTG printing requires hard alpha with no semi-transparency. A deterministic edge-refine
pass (`refine.py`) strips the garment-color halo along contours, reusing the same known-bg signal.

**Model & loss.** A lightweight background-conditioned U-Net (~2M params, `4→1` channels).
Loss = boundary-weighted BCE + L1 + a Laplacian-pyramid term for edge sharpness, with a mild
over-crop penalty (keep-real-art bias). Trains on a single GPU — including Apple-Silicon **MPS** —
in a few hours.

**Deployment shape.** Ship model-first behind a deterministic quality gate; fall back to the classic
cascade only on the hard tail (tonal collision), where the answer is genuinely ambiguous. You never
need "perfect on any graphic" — you need "confident on the majority, fall back on the rest."

## Garment-colored regions *inside* the art

What should happen to artwork regions that are the same (or nearly the same) color as the garment?
Classic DTG prepress answers this with geometry, not intent-guessing, built on one asymmetry:
**fabric is a perfect ink for the garment color, but ink is not.** A knocked-out region shows
fabric — a flawless color match by definition. The printed version of that color sits on a white
underbase (on darks), reads lighter with ink sheen, and stiffens the garment — the classic advice
is *"use the shirt as your black."* So removal is never wrong chromatically; its only failure mode
is structural (moth-eaten speckle). That reduces the policy to four rules:

1. **Connected to the outside background** (letter counters, gaps between elements) → always remove.
2. **At/near the garment color and bigger than the minimum printable feature** (~1 mm — below that,
   ink dot gain fills holes and the choked white underbase halos on misregistration) → remove;
   fabric renders it perfectly.
3. **Below the minimum feature size** → neither a hole nor a speck survives printing: fill enclosed
   micro-holes, drop isolated micro-specks (`cleanup.py`).
4. **Near-match embedded in continuous art** (dark shading on a black garment, gradients passing
   near the ground color) → keep; punching holes mid-gradient reads as damage, a slight tonal
   shift doesn't.

The model + `refine.py` handle rules 1, 2 and 4 (that's what the training distribution teaches);
`cleanup.py` enforces rule 3 deterministically on the final binary alpha.

One standing caveat: a knockout is **married to the garment color it was keyed against**. Removal
of garment-colored regions is lossless only on that garment — reuse the same cutout on a different
colorway and every knocked-out interior becomes a hole showing the new color. Re-run the knockout
per colorway.

## What it does well

On held-out data the model has never trained on, it:

- **Generalizes to unseen designs** — it learns the *task*, not the training images.
- **Holds up on graphic styles absent from training**, and improves further as style diversity is
  added to the foreground pool.
- **Keeps disjoint captions and logos** — the specific failure mode of salient-object segmenters.
- **Produces clean, binary, print-ready edges** with no semi-transparency.

The hard cases are where the artwork color is close to the garment color (tonal collision) or on
very dense/dark detail — inherently ambiguous inputs, where a deterministic fallback keyer is the
safety net. Higher training resolution reduces the residual over-crop.

## Repository layout

```
knockout/
  router.py            upstream gate: knockout vs full-bleed (should we remove the bg at all?)
  synth_composite.py   dataset generator: transparent PNGs → (input, alpha, bg_hex) composites
  base_colors.py       example garment-color distribution (sampling weights)
  dataset.py           loader: RGB + color-distance-to-bg channel → alpha
  model.py             background-conditioned matting U-Net (~2M params)
  losses.py            boundary-weighted BCE + L1 + Laplacian, over-crop penalty
  metrics.py           color-distance signal, deterministic boundary-residue gate, GT scores
  refine.py            deterministic edge decontamination (halo strip) on the model's alpha
  cleanup.py           minimum-feature cleanup: fill micro-holes, drop micro-specks
  train.py             training loop (MPS / CUDA / CPU)
  eval.py              held-out eval + deterministic gate
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build a dataset from a directory of transparent-PNG foregrounds
python -m knockout.synth_composite --fg-dir <foregrounds/> --out-dir data/train --n 20000

# 2. Train (MPS on Apple Silicon, CUDA on NVIDIA, else CPU)
python -m knockout.train --data data/train --epochs 14 --size 256 --device mps

# 3. Evaluate on a held-out set
python -m knockout.eval --data data/val --ckpt checkpoints/best.pt
```

At inference the model takes an image plus its known background hex, builds the color-distance
channel, runs one forward pass, and the edge-refine produces the final binary cutout.

## Data & weights

This repository contains **code only**. Training data (foreground corpora, generated composites)
are **not** included — they're large and largely third-party, and the label pipeline is
reproducible from any set of transparent PNGs via `synth_composite.py`.

**Weights** aren't published yet — this is a research checkpoint. The loader works with a local
export today, and with the Hugging Face Hub once weights are released (safetensors — no pickle):

```python
from knockout.pretrained import from_pretrained
model = from_pretrained("weights/v6")             # local export
# model = from_pretrained("<org>/dtg-knockout")   # once published to the Hub
```

Export a checkpoint with: `python -m knockout.pretrained --ckpt checkpoints/best.pt --out weights/vX`

## Status

Research prototype under active development. Evaluated on held-out synthetic composites and
real-domain generated designs, not yet a formal benchmark. Not production-blessed.

## License

[MIT](LICENSE) — free to use, modify, and ship (including commercially). All we ask is that you
keep the copyright and license notice (i.e. credit Algorithmic Labs).
