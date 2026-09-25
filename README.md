# Zero-shot fractional vegetation cover with multimodal large language models

Inference and analysis code for the paper *Exploring the Use of Multimodal Large
Language Models for Zero-Shot Fractional Vegetation Cover Estimation*
(submitted, *Ecological Informatics*).

Six open-weight multimodal models estimate fractional vegetation cover (FVC)
from 1,155 ground-level photographs of a 1 m x 1 m quadrat in arid rangelands of
northern Saudi Arabia. Nothing is fine-tuned and no model is shown a labelled
example: the task is specified entirely in natural language, and each model
returns a vegetation percentage and a self-reported confidence as JSON.

The photographs, the field reference cover, the annotations and every model
response are deposited separately on Zenodo:
**[10.5281/zenodo.22165601](https://doi.org/10.5281/zenodo.22165601)**.

## What the study measures

Six models by four prompt designs by three image variants, against every
photograph. Error is reported two ways throughout, because 933 of the 1,155
quadrats hold 20% cover or less and a single pooled statistic over data of that
composition describes the sparse end and little else. Overall MAE weights every
image equally; balanced MAE averages the five cover bins, giving sparse and dense
quadrats equal weight.

## Layout

| Path | What is in it |
|---|---|
| `inference/` | the code that generated the model responses, organised by serving arrangement |
| `parser/fvc_parser.py` | reduces one response to a vegetation percentage and a confidence |
| `notebooks/` | one jupytext notebook per research question, `Q0` to `Q9` |
| `results/` | every statistic the paper reports, one CSV per analysis |
| `PAPER_MAP.md` | each number in the paper traced to the file that produced it |
| `PROVENANCE.md` | how the results were produced, and where this code departs from the deposited tables |

### `inference/`

Three code paths were needed to reach all six models, because the serving
arrangement is what differs between them.

| Folder | Models | Hardware | Engine |
|---|---|---|---|
| `A100-Transformers/` | Gemma-3-12B, Gemma-3-27B, Mistral-Small-3.2, Qwen-2.5 | 1x A100-SXM4-80GB each | Hugging Face `transformers` |
| `H200-vLLM/` | Llama-4-Maverick, Llama-4-Scout | 2-4x H200 each | vLLM |
| `API/` | all six, through one gateway | hosted | OpenRouter |

Llama-4-Maverick and Llama-4-Scout could not run through the `transformers`
path. Maverick's FP8 checkpoint decompresses to about 803 GB in memory under
`transformers`, which no available allocation could hold, and Scout generates at
roughly 37,900 ms per image there against roughly 24 ms under vLLM. Each folder
has its own README.

### `notebooks/`

One notebook per research question, each standalone, each setting an explicit
seed in its first cell. Question identifiers thread through the whole
repository: `Q1` in `notebooks/` writes `results/Q1_*.csv`, and `PAPER_MAP.md`
ties those files to the sentences that cite them.

| Notebook | Question |
|---|---|
| `Q0_baseline_index.py` | the ExG-ExR vegetation-index baseline |
| `Q1_accuracy.py` | accuracy by model and prompt |
| `Q2_ensembles.py` | whether combining configurations helps |
| `Q3_bins.py` | error across the cover gradient |
| `Q4_preprocessing.py` | whether masking and rectifying the photograph helps |
| `Q5_confidence_part1_ranking.py`, `Q5_confidence_part2_discrimination.py` | whether self-reported confidence is usable |
| `Q6_device.py` | why a device effect cannot be separated from campaign |
| `Q7_serving_stack.py` | local deployment against the hosted gateway |
| `Q8_low_cover_bins.py` | behaviour inside the sparsest bin |
| `Q9_baseline_part1_pattern.py`, `Q9_baseline_part2_contrasts.py` | models against the classical baseline |

Notebooks are authored as jupytext percent-format `.py` files and rendered to
`.ipynb` under `notebooks/rendered/`. Edit the `.py`.

## Environment

`pixi.toml` and `pixi.lock` pin the exact versions that produced everything in
`results/`: Python 3.12.14, pandas 3.0.5, NumPy 2.5.2, SciPy 1.18.0,
statsmodels 0.15.0 and matplotlib 3.11.1.

```bash
pixi install
pixi run render notebooks/Q1_accuracy.py
```

## Input data

The notebooks read the deposited tables, not the photographs, except for `Q0`,
which computes the vegetation-index baseline from the rectified images.

The deposit stores one row per photograph and concatenates the model responses
into two files. The notebooks read the shape the analysis was written against,
so one script converts between them:

```bash
python scripts/from_zenodo.py --dataset path/to/unzipped/dataset --out data/raw
```

It checks the row counts and the cover bin composition before writing anything,
and it sets the row order deliberately, because a bootstrap draws from the
distinct images in the order the frame holds them. Run `Q0_baseline_index.py`
first: it writes the classical baseline the Q9 notebooks read, which is derived
from the rectified photographs rather than measured, and so is not deposited.

Every table in the deposit keys on a column named `image`, holding the
photograph filename, so the reference cover, the annotations and the model
responses join without renaming. The three field campaigns are `campaign_1`
(November 2024, 420 photographs), `campaign_2` (January 2025, 581) and
`campaign_3` (April 2025, 154), and this code uses the same names.

## Reading the numbers

Two things are worth knowing before comparing a figure here against the paper.

Cover bins are **right-closed**: a quadrat at exactly 20% falls in the `0-20`
bin. Twenty-five quadrats sit on a boundary, and binning them the other way
moves the sparsest bin from 933 photographs to 908.

Capture device is confounded with campaign. One phone captured campaigns 1 and
2, the other captured campaign 3, so no image exists that separates them.
`Q6_device.py` exists to demonstrate that rather than to estimate a device
effect, and reports the 2x3 table cell by cell instead of collapsing it.

`PROVENANCE.md` records the two places where this code knowingly differs from
the deposited tables, both in `Q9`, and neither affecting a number the paper
prints.

## Licence

Code is MIT (`LICENSE`). The deposited dataset is CC BY 4.0 and carries its own
terms. Each model is distributed under its own licence by its publisher; open
weights are not unencumbered weights, and those terms govern any use of the
models themselves.

## Citation

See `CITATION.cff`. Please cite the paper; if you use the photographs or the
responses, cite the Zenodo dataset as well.
