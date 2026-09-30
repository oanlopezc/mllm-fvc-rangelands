# Run report

Every notebook in `notebooks/` was run one at a time on a single machine. Each
run writes its CSVs to `results/`.

The table below records how long each question takes and how many result files
it writes, so a reader can size a re-run before starting one. Q1 is by far the
longest: its bootstrap and top-set machinery dominate the total.

| Question | Runtime | Result files in `results/` |
|---|---|---|
| Q0 | 2 min | 2 |
| Q1 | 83 min | 12 |
| Q2 | under 1 min | 5 |
| Q3 | 37 min | 9 |
| Q4 | 62 min | 8 |
| Q5 | 13 min | 16 |
| Q6 | under 1 min | 3 |
| Q7 | 4 min | 6 |
| Q8 | 11 min | 7 |
| Q9 | 3 min | 13 |

Q5 and Q9 are each split across two notebooks (`_part1_` and `_part2_`); the
runtime above covers both parts of the question.

Runtimes are wall clock on the machine used for the published run. They scale with core
count and with the bootstrap replicate count `B_BOOTSTRAP` in
`notebooks/_common.py`.
