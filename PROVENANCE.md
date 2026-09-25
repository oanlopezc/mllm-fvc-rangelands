# Provenance

How the numbers in `results/` were produced, and the two places where they
differ from the tables deposited with the paper.

## The chain

```
photographs + field reference cover        (deposited dataset)
   -> inference/           six models x four prompts x three image variants
   -> parser/fvc_parser.py one vegetation percentage and one confidence per response
   -> notebooks/Q*.py      one notebook per research question
   -> results/Q*.csv       every statistic the paper reports
```

Each notebook writes only to `results/`, sets an explicit seed in its first
cell, and resolves its input paths rather than assuming a working directory.
`PAPER_MAP.md` ties each file in `results/` to the sentences that cite it.

## Parsing happens once, afterwards

Every model response is written to a row log in full. The vegetation percentage
and confidence behind every published number come from running
`parser/fvc_parser.py` over those logs, not from anything a worker computed
while the grid was running. One parser therefore produces the local values and
the hosted-gateway values alike, so a difference between the two arms cannot be
an artefact of two different readers.

This matters for one comparison in particular. Section 3.7 sets the local
deployment against the hosted gateway, and that contrast is only interpretable
if both sides were read identically.

## The classical baseline is computed, not supplied

`Q0_baseline_index.py` computes the ExG-ExR baseline from the rectified
photographs under a fixed seed, and every Q9 comparison in this tree uses that
computation. It is reproducible from the deposited images alone.

## Where this code differs from the deposited tables

Two differences, both in Q9, neither affecting a number the paper prints.

**The deposited Q9 tables mix two versions of the baseline.** Eight of the ten
were produced against an earlier baseline whose overall MAE is 10.95, Pearson r
0.529 and mean bias -10.17. The baseline this code computes gives 10.92, 0.515
and -10.00, and those are the values the paper prints. This tree uses the
computed baseline throughout, so its Q9 outputs differ from those eight
deposited files by design. No value held by any of them appears in the
manuscript, which was checked against all thirteen source files.

**Per-bin confidence intervals are estimated on an isolated random stream.**
Drawing the per-bin bootstrap from a generator shared with upstream work makes
these intervals depend on how much bootstrapping happened earlier in the same
notebook, so adding any resampling above them silently moves them. Here the
per-bin table draws from its own generator. Point estimates are identical to the
deposited tables: the per-bin MAEs, the signed biases and Pearson r all match
exactly. Interval bounds differ by at most 0.24 cover points, which is Monte
Carlo noise at 10,000 draws rather than a change of result. Section 3.5 prints
`[70.62, 83.31]` for the 80-100% bin where this code gives `[70.85, 83.20]`.
Preserving the printed digits would have meant preserving the shared generator,
so the isolated stream was kept instead.

## Files in `results/` that the paper does not cite

`results/` holds every file the notebooks produce, which is more than the paper
uses. The campaign-clustered companions are the clearest case: they re-estimate
intervals treating the three field campaigns rather than individual images as
the sampling unit. With three campaigns a clustered interval carries two degrees
of freedom, which is too few to be informative, so they are reported here as a
sensitivity check and cited nowhere. `PAPER_MAP.md` marks which files the
manuscript draws on.
