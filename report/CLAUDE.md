# CLAUDE.md — Report

Rules for writing the project report. Source lives in `report/source/` (CVPR
two-column template, `cvpr.sty`). This file governs **only** the report.

## HARD CONSTRAINT — page limit
- **Content may use up to 4 full pages. References are separate and unlimited.**
  The bibliography may run for as many additional pages as needed (page 5+).
- "Content" = everything up to where the references begin (the
  `\bibliography{...}` / `thebibliography` block is the boundary). Content must
  not pass the bottom of page 4; references may.
- Fewer than 4 content pages is perfectly fine. Do not pad to fill space, but the
  full 4 pages are available if the content needs them.
- This is the top rule. Every other rule yields to it.

## Always compile after writing
- Every time you write or edit the report, **compile it to a PDF** at the end so
  the user can read the result. Do not finish a report edit without producing an
  up-to-date `main.pdf`.
- Compile from `report/source/`, e.g. `latexmk -pdf main.tex` (or `tectonic
  main.tex`). Report the output PDF path in your response.
- If no LaTeX compiler is installed locally, say so explicitly and offer to
  install one (tectonic is the simplest, single-binary option) before falling
  back to "compile on Overleaf". Do not silently skip the compile step.

## Verification protocol (run on EVERY content addition)
After adding or changing any content, before considering the edit done:
1. Recompile the report to PDF (`latexmk -pdf main.tex`, or Overleaf).
2. Confirm the references begin on or before page 4 and no content sits on
   page 5+.
3. If it overflows, **trim before finishing** — tighten prose, cut figures, or
   adjust layout. Do not leave the report in an over-length state.
4. State the resulting page count in the response (e.g. "4 pages + 1 ref page").

Quick page count once a PDF exists (any of):
- `pdfinfo main.pdf | grep Pages`
- `latexmk` log / Overleaf page indicator.

> Compiler: Tectonic is installed locally at
> `%LOCALAPPDATA%\tectonic\tectonic.exe` (not on PATH). Compile from
> `report/source/` with:
> `& "$env:LOCALAPPDATA\tectonic\tectonic.exe" -X compile main.tex`
> The page count appears in `main.log` as
> `Output written on main.xdv (N pages, ...)`.

## Cross-validate every claim with a separate agent
- Every factual claim, number, name, or finding written in the report MUST be
  cross-validated by a SEPARATE verification agent (Agent tool) against primary
  sources (code, configs, data, logs) before the edit is considered done. Do not
  rely on the writing pass alone.
- The verifier re-derives each claim independently and flags anything
  unsupported, approximated, stale, or wrong. It must not take the report's own
  wording as evidence.
- Fix or remove any claim the verifier cannot confirm from primary sources, then
  state in the response what was checked and the verdict.

## Reference configuration
- The model we actually trained and report on is the **stable fp32** config
  (`configs/resplan_housediff_stable_fp32.yaml`). Treat it as the single source
  of truth for every hyperparameter, precision, and batch-size claim in the
  report.
- Do NOT quote numbers from the default (`resplan_housediff_def.yaml`) or test
  configs as the reported model. If a value differs between configs, use the
  stable fp32 one.

## Conventions
- New rules go in this file as the report standards evolve.
- Keep prose tight; the 4-page limit is the binding constraint on scope.
