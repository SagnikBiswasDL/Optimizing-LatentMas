# ACL paper draft

Methods paper: **SynthScaffold** replaces sequential LatentMAS's silent
Planner/Critic/Refiner with one frozen **type-level KV prefix**.
The recipe depends on the problem type. The Judger still decodes.
Swap / probes live in the **appendix** (why a shared prefix is legal).
AIME is **not** a main-table result (long-decode batched harness
under-scores Real; Gate 1 at $B{=}1$ still matches the original paper).

Do **not** write the paper as a contradiction of Zou et al.: sequential
LatentMAS still beats a single agent; we skip upstream while matching Real.

## Build

Overleaf (this is the intended compiler): official **ACL** project,
**pdfLaTeX**, paste `main.tex`. The bib is embedded in `main.tex` and
overwrites `custom.bib` on compile. Then **Recompile from scratch**.
Do not use XeLaTeX/LuaLaTeX. Keep `acl.sty` and `acl_natbib.bst`.

```bash
cd paper/acl
python3 make_figures.py
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

## Figures

| File | Where |
|---|---|
| `figs/swap_budget` | Appendix: real vs shuffled vs none (motivation) |

Main numbers are tables. Eviction plots are unused.

## Locked numbers (Qwen3-14B, greedy, $K{=}10$)

| Setting | Recipe | $n$ | Real | Ours | None |
|---|---|---|---|---|---|
| MedQA $T{=}1024$, $B{=}20$ | Gaussian | 40 | **67.5** | **67.5** | 32.5 |
| GSM8K $T{=}1024$, $B{=}20$ | Mean-Replay | 40 | **90.0** | **90.0** | 80.0 |
| GSM8K same slice | Gaussian (transfer) | 40 | 90.0 | 82.5 | 80.0 |
| GSM8K $T{=}1024$, $B{=}20$ | KV-mean (failed) | 150 | 90.7 | **24.7** | 82.0 |

AIME24 (do not put in the main table): original paper sequential 14B
$66.7$ vs Single $63.3$; our Gate 1 `LatentMASMethod` $B{=}1$ $T{=}8192$
$K{=}10$ **$66.7$** vs $K{=}0$ $53.3$; synth harness $B{=}8$ dropped Real
to $33.3$ $=$ None — treat as a broken Real, not a finding.

Original LatentMAS sequential 14B (sampled, full sets): GSM8K $95.2$ vs
Single $83.7$; MedQA $80.7$ vs $64.7$. Our $T{=}4096$ MedQA Real $80.0$
on $n{=}40$ is in that band.
