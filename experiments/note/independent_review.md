# Independent closure review of the TAFR experiment note

**CLEAN: 0 critical, 0 major, 0 minor surviving findings for the reviewed note snapshot.**

Scope: the internal preliminary research note and article outline, including number macros, four generated result tables, the comparison table, and eight included data figures. Controlling idea: **Finite-radius knee claims require separate geometry, optimizer-map, and gain checks.** The reviewer independently reran the mechanical gate and applied the accessibility, recursive-followability, lexical/decomposition, grounding, and positioning checks. The author’s claim of having fixed an item did not serve as evidence; the current text, source code, and raw records did.

Reviewed note SHA-256: `2e3fd0cd3ebfd854b53d2a36d0974a26f9d910798147e458afbc062b3bde849b`.

The final-source comparison shows only one change from the preceding CLEAN snapshot: the capability table now uses ragged-right X columns. Substantive prose is unchanged. The final mechanical rerun has the same counts and justified false positives.

`validation_extensions.json` now contains 99 completed extension records: 84 exact finite-table PMOP baseline audits, six exact finite-table discrete baseline audits, and nine continuous budget extensions. Direct record comparison confirms that **all nine continuous R values equal their previous values exactly**. Every extended sample count is at least ten times its recorded total training-audit point count: 590/59 for the three SNEE extensions, and 270/27, 300/30, or 320/32 for the six ammonia extensions. The LA global-bound TAFR value remains 0.017163349715426615. Thus “at least 256 new perturbations” remains accurate. The final table/figure regeneration and PDF rebuild remain the author’s delivery step.

## Numbered finding closure

1. **CLOSED — S1/S2/S3/S4, line 39 and comparison table lines 43–51.** RI and AWT now expand before their first use in the comparison table; the detailed persistence and scalarization definitions remain in their home sections.

2. **CLOSED — S1/S4/S11, lines 121–132.** The note gives both shielding vectors and defines `Y` as the seven objective rows. The vectors agree with `adversarial_tabular.interior_cell`, and Figure D’s R1/R2 labels now have referents.

3. **CLOSED — S1/S4/S15, lines 223–227.** The note defines the relative simplex neighborhood, the alpha grid from 0 to 1 in steps of 0.2, and the actual draw counts. Source verification confirms 32 per level for TAFR post-analysis (`run_pilots.py`) and **256** per level for the portfolio AWT diagnostic (`run_scalability.py`), a distinction the revision correctly includes. The normalization interval has length one.

4. **CLOSED — M11, line 348.** The bibliography now uses active voice: “this release excludes.” Independent passive grep: zero hits.

5. **CLOSED — S24/S28, lines 115–118 / Figure C.** The prose explicitly maps red to +0.1, blue to -0.1, and gray to zero. Figure values, axes, and legend context are legible.

6. **CLOSED — S17/S18, line 93.** The note now calls `[0.4,0.6]` K’s co-optimality interval. This distinguishes scalarized ties from the deterministic tie rule while preserving the correct stability supremum of 0.1.

7. **CLOSED — S8/S9, lines 31, 142, 201, 305, and table_pmop.tex:5.** The prose consistently uses “TAFR gain variant” and “optimizer map.” The old descriptive synonym no longer appears in the note. The `TAFR-meaningful-gain` raw-record identifier remains a code identifier in source/data, not a competing prose concept.

8. **CLOSED — S4/S17, line 115.** The note separates the perturbation delta from the feasible weight `(0.35,0.35,0.15,0.15)`. The raw four-objective example confirms that this weight selects Q.

9. **CLOSED — S18, line 261.** Nested samples now restore only sampled monotonicity. Exact stability explicitly requires a global oracle or a verified adversarial solver.

10. **CLOSED — S1/S4/S15, lines 196–199.** The note defines normalized Euclidean nearest-knee error and Success@0.025, including unsuccessful abstention. These definitions agree with `knee_matching.metrics` and the table aggregation.

11. **CLOSED — S4/S15, lines 161–175.** The note distinguishes the original-start NM/DIRECT experiment from the NM-only 540-job multi-start experiment. All 18 table aggregate groups agree with raw MCF quantiles within `2.22e-16`.

12. **CLOSED — S17/S18, lines 291–293.** The water/safety dependence claim now applies to the fixed-output slice attained by the retained schedules. Direct summation confirms 45,000 kg NH3 for all nine retained schedules across the two windows and two normalization variants. The model itself still has a lower bound, which the revision explicitly states.

13. **CLOSED — S1/S4, lines 130–132.** Hybrid and Adaptive auditing now have operational definitions that agree with the implemented vertices/transfers/interior sampling and adaptive sampling from the current worst point.

14. **CLOSED — S3/S15, lines 27–28, 298, 302, 315, and bibliography.** The note expands TPE, cites its source, expands the two venue names, and defines RVEA/KnEA with a source for the population algorithms. Published software and benchmark names remain names; the note does not invent expansions for them.

## Final mechanical result

| Category | Raw grep output lines | Disposition |
|---|---:|---|
| M1 dash forms, including extra spaced-en-dash scan | 0 | Clean |
| M11 passive voice | 0 | Clean |
| M2 factual negations | 26 | All justified below |
| M3/M4/M8/M9 filler; extra M4 listed words | 0 | Clean |
| M6 openers | 0 | Clean |
| M5/M16 words | 2 | Feature-file noun and external title |
| M10 vague verbs | 0 | Clean |
| M17 fancy verbs | 0 | Clean |
| M18 content-free openers | 0 | Clean |
| M12 wordiness | 0 | Clean |
| M13 weak qualifiers | 0 | Clean |
| M14 adverbs | 7 | Six `pairwise`, one `piecewise` |
| M15 exclamation | 2 | LaTeX `resizebox` arguments |
| Precision pairs | 0 | Clean |
| S9 decomposition cardinality | 0 | No conflicting counted decomposition |
| S1 CamelCase aid | 9 | Macro, software, and already-defined algorithm names |
| S8 canonical-term aid | 41 | Every line inspected; no surviving drift |

All 26 M2 hits preserve a factual limitation, excluded process, mathematical failure, or scope boundary. The two “not a” contrasts delimit what the grouping example and ammonia pilot establish. Removing those negations would remove scientific content.

The M5/M16 hits are `feature` at line 274, naming input feature files, and `Novel` at line 346, inside the exact title of an external proposal. Neither is promotional prose authored for this note. M14 hits are mathematical terms, and M15 hits are TeX syntax. Macro names, SciPy, HiGHS, NumPy, LaTeX, PlatEMO, and the first-use-expanded KnEA justify the S1 grep aid. M7’s geometry/optimizer-map/gain triad contains three distinct necessary checks; no decorative triad remains.

## Semantic closure

The thesis remains fixed, and each experiment addresses geometry, the optimizer map, gain screening, or the limits on interpreting those checks. The text separates severity from persistence, exact finite-table values from sampled continuous lower bounds, selection from front generation, supported from unsupported knees, runtime failures from mathematical inapplicability, and withheld conclusions from completed observations. The article outline assigns unresolved experiments to future gates without presenting them as completed contributions.

No new undefined load-bearing construct, conflicting decomposition, illogical implication, ungrounded table entry, or unscoped superiority claim survived the final fresh-reader pass. This is an internal note, so final venue submission and significance requirements remain explicit future work rather than an unfulfilled claim in the current deliverable.

## Numerical grounding and visual evidence


- `numbers.tex` maps to `make_note_assets.main` and `adversarial_phase1.json`. Checked: pairwise R=0; four-objective exact/vertex R=0.208806130178211; interior-cell vertex R=0.17883362798981629 versus exact R=0.3; stability radius 0.1; exit ratio 1.5; and 43/100 versus 97/100 detection at 16 and 128 samples.
- All PMOP summary rows agree with `pmop_pilot_table_range.json`: TAFR returns 36/42 with 18 successes, equal weights 42/42 with 26, and TPE 42/42 with 17. Recomputed error quartiles reproduce the rendered rounding. PMOP12 with five objectives records reported R=0.07159180209413861 and oracle R=0.07358692015636702; it is the single TAFR understatement above `1e-7`.
- `snee_reproduction_v2.json` contains 45 completed and nine failed jobs with exactly the stated failure identities. `snee_multistart_v2.json` contains 540 completed NM jobs. All table values trace through `aggregate_results.main` and `make_note_assets.main`.
- Each discrete table row agrees with `discrete_pilot.json`: feasible counts 8633, 8155, 8566; exact R=0 in all three; exit ratios 1.0857890282, 1.6927123552, 1.0621118012; AWT equal-weight RI 0.553125, 0.6171875, 0.398046875.
- All ammonia rows map directly to the two named raw bundles. The LA global-bound point has reported and independently sampled R=0.017163349715426615, exit ratio 1.5081407908627191, and estimated stability radius zero. The note correctly identifies invalid near-zero payoff ranges, mixed changes to normalization/budget, fixed forecasts, and the sampled rather than exact validation status.
- The main claims stay tied to geometry, optimizer-map, and gain checks. The pilots do not support algorithmic superiority or continuous certification, and the note says so. Different scalarizations, supported versus unsupported knees, abstention, and finite-table versus full-decision tracks remain distinct. The repeated four-objective displacement in the reduction example applies the same counterexample to a second necessary check; it is a justified reference rather than a duplicated empirical contribution.
- Standalone visual inspection covered all eight included data figures: A, B, C, D cells, D detection, E, audit validity, and scalability. Labels and plots are legible. The revision now explains the Figure C colors and names the Figure D shielding rows. A read-only render of the available 12-page PDF also shows no clipped tables, overlaps, or broken figures. That PDF predates the latest naming edits, so this visual check does not replace the author's final rebuild.
- Review scope is an internal preliminary note and outline. Pending final experiments are explicit project gates rather than falsely completed requirements. Venue-specific submission thresholds (S30) do not gate this internal document.


## Independent raw grep evidence: final closure run

### M1: Em-dashes (must return nothing)

Count: 0 matching output lines.

```bash
grep -rn -- '---' experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex ; grep -rn $'—\|–' experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M11: PASSIVE VOICE (also runs in the base per-edit gate). Over-catches on purpose.

Count: 0 matching output lines.

```bash
grep -rnoE "\b(is|are|was|were|be|been|being)\s+([a-z]+ed|done|made|shown|given|taken|held|built|drawn|chosen|written|known|found|seen|set|put|sent|kept|met|run|used|based)\b" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M2: Antithesis / negation-as-rhetoric (inspect each; "not" branch matches digit/letter/\macro)

Count: 26 matching output lines.

```bash
grep -rnoE ",? not [0-9A-Za-z\\]|not only .* but|rather than|less .* than|is the point|whatever it is|means nothing|more than just|not in competition with|on one hand|on the other hand" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
experiments/note/TAFR_Experiment_Note.tex:18: not e
experiments/note/TAFR_Experiment_Note.tex:29: not r
experiments/note/TAFR_Experiment_Note.tex:79: not m
experiments/note/TAFR_Experiment_Note.tex:86: not r
experiments/note/TAFR_Experiment_Note.tex:96: not i
experiments/note/TAFR_Experiment_Note.tex:98: not i
experiments/note/TAFR_Experiment_Note.tex:120: not m
experiments/note/TAFR_Experiment_Note.tex:133: not a
experiments/note/TAFR_Experiment_Note.tex:151: not g
experiments/note/TAFR_Experiment_Note.tex:158: not e
experiments/note/TAFR_Experiment_Note.tex:159: not f
experiments/note/TAFR_Experiment_Note.tex:165: not e
experiments/note/TAFR_Experiment_Note.tex:192: not d
experiments/note/TAFR_Experiment_Note.tex:208: not i
experiments/note/TAFR_Experiment_Note.tex:211: not e
experiments/note/TAFR_Experiment_Note.tex:226: not e
experiments/note/TAFR_Experiment_Note.tex:233: not e
experiments/note/TAFR_Experiment_Note.tex:246: not m
experiments/note/TAFR_Experiment_Note.tex:248: not e
experiments/note/TAFR_Experiment_Note.tex:255:, not a
experiments/note/TAFR_Experiment_Note.tex:261: not p
experiments/note/TAFR_Experiment_Note.tex:286: not e
experiments/note/TAFR_Experiment_Note.tex:288:, not a
experiments/note/TAFR_Experiment_Note.tex:294: not i
experiments/note/TAFR_Experiment_Note.tex:294: not s
experiments/note/TAFR_Experiment_Note.tex:301: not e
```

### M3/M4/M8/M9: editorializing, intensifiers, grandiose setups, metaphor filler

Count: 0 matching output lines.

```bash
grep -rnoE "in effect|in a sense|at (its|the) (heart|core)|in essence|\btruly\b|\bgenuinely\b|\bindeed\b|\bin fact\b|precisely because|a testament to|the kind of .* that|exactly the kind|is the point|set(s)? .* apart|no (predecessor|one) .* (made|posed)|the key (insight|idea) is|the machine that|draw(s)? .* power from|under the hood|where .* meets" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M6: throat-clearing openers

Count: 0 matching output lines.

```bash
grep -rnE "^(Moreover|Furthermore|Additionally|Notably|Importantly|Indeed|Ultimately|Crucially|In turn|That said)" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M5/M16: banned + pompous words

Count: 2 matching output lines.

```bash
grep -rnoiE "\bnovel\b|\bsignificant\b|\bsubstantial\b|\bimpressive\b|\bpromising\b|\bcomprehensive\b|\brobust\b|\bpowerful\b|\bseamless|\bcrucial\b|\bparadigm\b|\bleverag|\butiliz|\bfinaliz|[a-z]+-oriented\b|\bfactor\b|\bfeature[ds]?\b|\bmeaningful\b|\binsightful\b|\bprestigious\b|\bpossess|\bcontact(s|ed|ing)?\b|\bcurrently\b|\bimpact(s|ed|ing)?\b" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
experiments/note/TAFR_Experiment_Note.tex:274:feature
experiments/note/TAFR_Experiment_Note.tex:346:Novel
```

### M10: vague-mechanism / futurist hype verbs

Count: 0 matching output lines.

```bash
grep -rnoiE "promises to|stands? to|is poised to|opens the door to|is set to|has the potential to|keeps .* from|stands? in the way|\bunlocks?\b" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M17: fancy / figurative verbs (inspect each; keep only precise domain jargon like "amortize")

Count: 0 matching output lines.

```bash
grep -rnoiE "\b(pit(s|ted|ting)?|dispatch(es|ed|ing)?|chip(s|ped|ping)? (away )?at|marshal(s|led|ling)?|orchestrat(e|es|ed|ing)|wrangl(e|es|ed|ing)|harness(es|ed|ing)?|forge[sd]?|weav(e|es|ed|ing)|delv(e|es|ed|ing) into|usher(s|ed)? in|grappl(e|es|ed|ing) with|anew|afresh)\b" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M18: content-free openers

Count: 0 matching output lines.

```bash
grep -rnE "In this (paper|section), we" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M12: needless words / wordiness

Count: 0 matching output lines.

```bash
grep -rnoE "the fact that|the question (as to |of )?whether|as to whether|in order to|there is no doubt but|the reason .* is because|owing to the fact that|in a [a-z]+ manner|is a (subject|man|woman) (that|who)|in the last analysis|along these lines|in terms of|one of the most" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M13: weak qualifiers

Count: 0 matching output lines.

```bash
grep -rnoiE "\b(rather|very|pretty|little|quite|somewhat|fairly|certainly)\b" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M14: coined adverbs / false ordinals

Count: 7 matching output lines.

```bash
grep -rnoiE "\b(thusly|muchly|overly|firstly|secondly|thirdly)\b|[a-z]+wise\b" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
experiments/note/TAFR_Experiment_Note.tex:15:pairwise
experiments/note/TAFR_Experiment_Note.tex:28:pairwise
experiments/note/TAFR_Experiment_Note.tex:112:pairwise
experiments/note/TAFR_Experiment_Note.tex:114:pairwise
experiments/note/TAFR_Experiment_Note.tex:116:Pairwise
experiments/note/TAFR_Experiment_Note.tex:130:pairwise
experiments/note/TAFR_Experiment_Note.tex:149:piecewise
```

### M15: exclamation marks

Count: 2 matching output lines.

```bash
grep -rn "!" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
experiments/note/TAFR_Experiment_Note.tex:234:\begin{table}[htbp]\centering\small\resizebox{\linewidth}{!}{\input{table_discrete}}
experiments/note/TAFR_Experiment_Note.tex:281:\begin{table}[htbp]\centering\scriptsize\resizebox{\linewidth}{!}{\input{table_ammonia}}
```

### Part B: precision pairs (inspect for wrong member)

Count: 0 matching output lines.

```bash
grep -rnoiE "\bcomprised of\b|\bdata is\b|different than|\bvery unique\b|\bdue to\b|\bless (than )?[0-9]" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### Term/decomposition drift (see gate_semantic S8/S9): one name per concept

Count: 0 matching output lines.

```bash
grep -rnoE "\b(three|four|five|six|seven)\b (stages|concerns|axes|requirements|dimensions|systems)" experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M1 spaced en dash

Count: 0 matching output lines.

```bash
grep -rniE ' -- ' experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### M4 additional listed words

Count: 0 matching output lines.

```bash
grep -rniE '\b(simply|essentially|fundamentally|real|really|actually|exactly)\b|armed with' experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
(no hits)
```

### S1 coined terms

Count: 9 matching output lines.

```bash
grep -rnE '[A-Z][a-zA-Z]+×[A-Z][a-zA-Z]+|\b[A-Z][a-z]+[A-Z][a-zA-Z]+\b' experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
experiments/note/TAFR_Experiment_Note.tex:129:Across 100 seeds, 16 samples detect the exact worst-case row in $\DetectSixteen$ runs, and 128 samples detect it in $\DetectOneTwoEight$ runs.
experiments/note/TAFR_Experiment_Note.tex:163:Variants retain published objective scales, adapt constrained gradient shapes for SciPy, or additionally normalize objectives using frozen payoff bounds.
experiments/note/TAFR_Experiment_Note.tex:267:HiGHS solves each mixed-integer linear program with relative gap $10^{-3}$ and a ten-second limit.
experiments/note/TAFR_Experiment_Note.tex:315:Second, validate the PMOP port numerically and implement the full-decision track with a reference vector guided evolutionary algorithm (RVEA), knee-selection rules, a knee point driven evolutionary algorithm (KnEA), and a valid geometric baseline. The pinned PlatEMO platform supplies these population algorithms~\cite{platemo}.
experiments/note/TAFR_Experiment_Note.tex:323:The runtime uses Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0, and bundled HiGHS 1.8.0 on Linux with an AMD EPYC 9V74 host CPU.
experiments/note/TAFR_Experiment_Note.tex:341:Artifacts include raw JSON, YAML configurations, processed CSV, vector figures, LaTeX tables, execution status, and a claim ledger.
experiments/note/TAFR_Experiment_Note.tex:350:PlatEMO: A MATLAB platform for evolutionary multi-objective optimization.
experiments/note/numbers.tex:7:\newcommand{\DetectSixteen}{43}
experiments/note/numbers.tex:8:\newcommand{\DetectOneTwoEight}{97}
```

### S8 canonical terms

Count: 41 matching output lines.

```bash
grep -rniE 'optimizer.map|solver map|displacement|severity|stability|persistence|gain variant|meaningful-gain|screen-passed|certified' experiments/note/TAFR_Experiment_Note.tex experiments/note/numbers.tex experiments/note/table_*.tex
```

```text
experiments/note/TAFR_Experiment_Note.tex:13:Finite-radius knee claims require separate geometry, optimizer-map, and gain checks.
experiments/note/TAFR_Experiment_Note.tex:21:\section{Preference stability needs separate checks}
experiments/note/TAFR_Experiment_Note.tex:28:We preserve that starting point through a TPE-2024-grid reconstruction that minimizes raw squared objective displacement under pairwise perturbations.
experiments/note/TAFR_Experiment_Note.tex:31:The TAFR gain variant raises both absolute exit-gain thresholds to $10^{-3}$ through the existing configuration interface.
experiments/note/TAFR_Experiment_Note.tex:45:TAFR & Candidate weights & Solver dependent & Sampled severity; external exact table oracle & Via callback \\
experiments/note/TAFR_Experiment_Note.tex:47:Mavrotas RI & Selected Pareto point & Solver dependent & Sampled persistence & Yes, including AWT \\
experiments/note/TAFR_Experiment_Note.tex:62:Its finite-radius displacement is
experiments/note/TAFR_Experiment_Note.tex:67:Displacement measures worst-case severity; the probability of retaining a solution measures a different property.
experiments/note/TAFR_Experiment_Note.tex:78:The stability oracle minimizes the $L_\infty$ distance from $w_0$ to each reachable competing cell's closure.
experiments/note/TAFR_Experiment_Note.tex:80:An objective tolerance $\epsilon$ excludes competitors with displacement at most $\epsilon$.
experiments/note/TAFR_Experiment_Note.tex:84:The production field \texttt{certified} has narrower semantics than a global mathematical certificate.
experiments/note/TAFR_Experiment_Note.tex:85:It checks distance from fitted anchors and an active exit ratio, then ranks accepted candidates by reported displacement and stability radius.
experiments/note/TAFR_Experiment_Note.tex:87:We therefore call a returned pilot point ``screen-passed'' and reserve ``exact'' for finite-table oracle results.
experiments/note/TAFR_Experiment_Note.tex:92:The table $A=(0,10)$, $K=(4,4)$, $B=(10,0)$ separates plateau stability from compromise quality.
experiments/note/TAFR_Experiment_Note.tex:94:At equal weights, the exact stability radius is $\RadiusA$; a normalized exit to either anchor has improvement $0.4$, deterioration $0.6$, and ratio $\ExitA$.
experiments/note/TAFR_Experiment_Note.tex:98:\caption{Plateau stability alone does not identify the interior compromise.}\end{figure}
experiments/note/TAFR_Experiment_Note.tex:120:Complete weight vertices need not maximize displacement through an optimizer map.
experiments/note/TAFR_Experiment_Note.tex:133:Its interior witness also exceeds its vertex displacement, but that witness is not an exact continuous maximum.
experiments/note/TAFR_Experiment_Note.tex:142:The TAFR gain variant requires $I,D\ge10^{-3}$; Figure~\ref{fig:E} also sweeps $10^{-4}$, $5\times10^{-3}$, and $10^{-2}$.
experiments/note/TAFR_Experiment_Note.tex:201:Equal normalized weights outperform the TAFR gain variant on Success@0.025 in this pilot.
experiments/note/TAFR_Experiment_Note.tex:214:One of 36 reports understates displacement beyond $10^{-7}$: PMOP12 with five objectives reports about $0.07159$, while the oracle gives about $0.07359$.
experiments/note/TAFR_Experiment_Note.tex:219:\caption{Independent validation can exceed reported optimizer-map displacement.}\label{fig:validity}\end{figure}
experiments/note/TAFR_Experiment_Note.tex:221:\section{Persistence differs from severity}
experiments/note/TAFR_Experiment_Note.tex:222:Mavrotas and coauthors analyze weight robustness through expanding relative neighborhoods and solution persistence~\cite{mavrotas}.
experiments/note/TAFR_Experiment_Note.tex:223:Our Mavrotas-style diagnostic jointly samples the relative box intersected with the simplex and computes the normalized trapezoidal area under the persistence curve, RI.
experiments/note/TAFR_Experiment_Note.tex:226:The continuous diagnostic declares persistence when normalized objective displacement does not exceed $10^{-3}$, an explicit experimental tolerance.
experiments/note/TAFR_Experiment_Note.tex:231:TAFR returns a point with exact displacement zero at radius $0.05$ in each instance.
experiments/note/TAFR_Experiment_Note.tex:232:Table~\ref{tab:discrete} also reports the equal-weight AWT persistence diagnostic.
experiments/note/TAFR_Experiment_Note.tex:233:Zero displacement on a plateau does not establish the best trade-off geometry or greatest RI; direct TAFR-AWT search remains pending.
experiments/note/TAFR_Experiment_Note.tex:235:\caption{Exact portfolio pilot and a separately defined AWT persistence diagnostic.}\label{tab:discrete}\end{table}
experiments/note/TAFR_Experiment_Note.tex:250:Grouping can hide original-space displacement.
experiments/note/TAFR_Experiment_Note.tex:252:The original domain still contains the switch to $Q$ with displacement $\RfullC$.
experiments/note/TAFR_Experiment_Note.tex:260:Production radius bisection also resamples at each radius, which can make estimated displacement nonmonotone.
experiments/note/TAFR_Experiment_Note.tex:261:Nested samples restore monotonicity of the sampled estimate but do not prove exact stability. An exact stability claim needs a global oracle or a verified adversarial solver.
experiments/note/TAFR_Experiment_Note.tex:277:ISO-NE water and safety anchor ranges lie near machine precision, making normalized displacement numerically unstable.
experiments/note/TAFR_Experiment_Note.tex:285:Independent sampled displacement is about $0.01716$, matching reported displacement, with exit ratio about $1.508$.
experiments/note/TAFR_Experiment_Note.tex:299:Mavrotas-style persistence is the closest conceptual comparison and belongs in the main related-work table.
experiments/note/TAFR_Experiment_Note.tex:300:The research direction is direct preference search with severity and exit checks, followed by independent auditing.
experiments/note/TAFR_Experiment_Note.tex:305:\item \textbf{Finite-radius formulation.} Define normalization, the optimizer map, severity, stability, absolute gains, and abstention, including tie and solver-accuracy assumptions.
experiments/note/TAFR_Experiment_Note.tex:309:\item \textbf{Discrete and scheduling decisions.} Compare weighted sums with AWT, severity with persistence, and the physical decisions behind selected schedules.
experiments/note/table_pmop.tex:5:TAFR gain variant & 36/42 & 0.0234 [0.0000, 0.2124] & 18/42 \\
```
