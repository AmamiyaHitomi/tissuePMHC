# HumanPMTC Premium B-E III seed analysis of results

Analysis date: 2026-07-31

Uniform agreements: premium train-only, fixed 3-fold plain-grouped OFF, 25 epochs,
Seeds ZXQ0QZ. All models have limited fractions and are located in `[0,1]`.
Runs time-consumingly to the terminal log and is not stored in the experimental outcome document.

## B: Assisted mission diagnosis

### B1 Positive and negative differences

Three seed, three fold, and then summarized:

| label | tissue accuracy | query tissue probability | tissue NLL | HLA accuracy |
|---:|---:|---:|---:|---:|
| 0 | 0.370313 | 0.270777 | 1.649659 | 0.600910 |
| 1 | 0.414692 | 0.292351 | 1.496266 | 0.774884 |

Negative case/ HLA auxiliary is clearly more difficult to support "query probleme lack of negative case peptide"
Supported diagnosis. As `other_tissue_count` increases, the ticue accuracy drops further.

### B2 Gradient Conflict

| label | main–tissue weighted cosine | tissue conflict fraction | main–HLA weighted cosine | HLA conflict fraction |
|---:|---:|---:|---:|---:|
| 0 | -0.048064 | 0.675214 | -0.147277 | 0.927350 |
| 1 | +0.047443 | 0.329060 | +0.148079 | 0.072650 |

The negative case main is more in conflict with the auxiliary gradient and HLA conflicts are more than tissue
Conflict is stronger. This is a mechanism diagnosis, not equivalent to deleting auxiliary would improve final performance.

### B3 tissue-label shuffle

The average change after the tissue auxiliary, fold is disrupted:

Indicators
|---|---:|
| mean task AUROC | -0.000491 |
| mean task AUPRC | -0.000105 |
| mean task PairAcc | +0.001032 |
| mean task MCC | -0.000764 |

Change is near zero, and mission victory is close to 50/50. So the exact label semantics of the Tissue Auxiliary
Not the main source of current performance; it's more like a regular signal. The gradient conflict of B2 exists.
But it has not been converted to "moderately changed performance after disrupting the tab."

## C: Conditional structure

### III Seed Mean

AUROC  AUPRC  PairAcc  MCC  work-10 AUROC
|---|---:|---:|---:|---:|---:|
| C0 | 0.676489 | 0.658684 | 0.678689 | 0.261562 | 0.520727 |
| C1 | 0.676574 | 0.659261 | 0.681228 | 0.261408 | 0.534830 |
| C2 | 0.681516 | 0.661815 | 0.686922 | 0.270728 | **0.538574** |
| C3 | 0.681446 | 0.662249 | 0.685361 | 0.270936 | 0.529718 |
| C4 | **0.685377** | **0.666096** | **0.690374** | **0.275577** | 0.536561 |

C2, C3 all meet the promotion requirements relative to C0; C4 is the best structure in the whole. C4 relative to C0:

- 66.7 per cent of the total number of references to AUROC ZXQ0QZ, 225, and the comparison of the ED-task;
- AUPRC ZXQ0QZ, 60.0% positive;
- PairAcc ZXQ0QZ, 60.4% positive;
- MCC ZXQ0QZ, 61.8% positive;
- The AUROC, AUPRC, PairAcc above are positive in all three seeds.

C4 is still available on the full unseen layer AUROC ZXQ0QZ, AUPRC ZXQ1QZ,
PairAcc ZXQ0QZ, no systematic degradation; the main gains are still concentrated twice in the little
. The AUROC and PairAcc of C4 are in 14 organizations by the tissue macro average
There were 10 upgrades; kidney was the main degradation organization, AUROC dropped `0.038347`.

C3 has 9,675 tab parameters; removing them to C4 and thus performing better.
The task re-development is more like a composite source at the current data scale, recommending C4 instead of C3.

## D: Whether the model relies on condition input

### D2 validation tissue shuffle

Seed average, relative to each other:

♪ The world's greatest ever ♪
|---|---:|---:|---:|---:|
| C0 | 0 | 0 | 0 | 0 |
| C2 | -0.014002 | -0.013572 | -0.012401 | -0.021920 |
| C3 | -0.004841 | -0.005319 | -0.004200 | -0.004472 |

C0 does not read the Tissue ID, the result is strictly unchanged. C2/C3 drops in all three seeds, which means both
True reliance on the Tissue input; C2 is clearly more dependent than C3.

### D3 Component Close

III Seed Average AUROC Change:

♪ The world is a place where you can't be seen ♪
|---|---:|---:|---:|---:|
C0 Not applicable  Not applicable  not applicable  -0.010679
C2 -0.012316
| C3 | -0.003393 | -0.012624 | -0.003775 | -0.009673 |

HLA conditions contribute more than the Tissue condition. `auxiliary off` closes the Tissue and HLA simultaneously
Auxiliary, so the decline cannot be attributed separately to the tissue auxiliary; more likely to be combined with B3
The HLA auxiliary or overall regular contribution.
But retrain and structurally remove the resocial C4 better, which means it's a common adaptation effect.
Can't deny C4.

### D1 tissue swap

The same rate as the real observed-issue:

♪ The way you see it ♪
|---|---:|
| C0 | 0.734264 |
| C2 | **0.763018** |
| C3 | 0.729060 |

C0 The high consistency rate is not evidence of organizational mechanisms because the switch to task head itself changes scores.
C2 is about 2.88 percentage points higher than C0 and is simultaneously D2 shuffle and D3 tessue-off,
This can be explained by the actual use of the Tissue condition. The swap consistency rate for C3 is not above C0, but D2/D3
The blogger also points out that the government is still in a position to prove that it has weak conditions.

D Only the condition can be demonstrated and the tissue-specific biological mechanism cannot be demonstrated separately.

## E: Processing-first new code completed and not forced to be fully operational

The examination found that the target tissue, HLA and parent UniProt of premium were identical for each positive or negative pair.
So, the parent expression alone cannot distinguish directly between the inside and the inside of the pair.
Deleted and replaced with E0-E4 design-first: first extract the real N/C using the parent level
From the front and the processing code, test the message from the missing person.
Interactive; interpretation-only retained as a negative contrast between expectations.

The new version code has completed the full process synthesis input, and the synthesis results are deleted. Full operation still needs to be performed
Real-time FASTA, UniProt-to-Ensembl map, HPA expression matrix and tissue map manually reviewed.
Bone, brain, lymphoid or uniical cod blind is not shown as a approximation organization without review.

## Final judgment

1. Negative examples Auxiliary gradient conflicts are clear, but the exact semantic terms of the tissue label have little effect on final performance.
2. Allows Tissue/ HLA entry to the master sign that it is valid and that the best structure is C4 without task rehabilitation.
3. C2 Reliance on the evidence is strongest for the conditions of the Tissue; C3 is weaker.
4. Current evidence supports "statistical conditions are effective" and does not yet support "organizing/expression mechanisms are effective".
5. The next step should be to complete the profile and artificial tissue mapping of the parent peptide, and then run the new version E on a single basis;
   Unvenered approximation features are not used.
