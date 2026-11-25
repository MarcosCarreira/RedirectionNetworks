# Complete redirection trees: the growth exponent (2026)

Marcos Costa Santos Carreira, with AI assistance (Claude, Anthropic) in code and checks. Not peer reviewed.

Paper: [`paper/CavityExponent_2026-11-25.pdf`](paper/CavityExponent_2026-11-25.pdf) (LaTeX source alongside), *The non-leaf exponent of complete redirection trees: cavity equations on degree paths*. It contains the proofs, the solver and the results summarised below, and in its appendices the exact one-step relations, the models DAN, PAN and VAN, the fresh-child closure, exact results for small trees and the 2022-2023 work.

## The question

In complete redirection, each new node picks an existing node uniformly at random and attaches to a uniformly chosen neighbour of it. The number of non-leaf nodes grows as N^mu. The reference value is mu = 0.565723 (s.e. 0.000003), from Uzay's simulations to 8e13 nodes. Uzay also proves that the exponent exists and, with computer assistance, that 0.565713486 <= mu <= 0.565736987 (github.com/uzay-eth/friend-tree-exponent, Round 3, 25 November 2026); C. D. Long characterizes mu through an inert newborn and proves that it is computable (github.com/long-mathematics/friend-tree-growth). These results are checked by AI models, not by human referees.

## Results

| Method | mu | against 0.565723 | Script and result file |
|---|---|---|---|
| Euler-Lotka equation with equal sharing of credit among non-leaf neighbours (deterministic closure) | 0.568842 | +0.0031 | `closure_solver.py`, `closure_solver_2026-11-22_225529.txt` |
| Euler-Lotka equation with the credit by state measured in simulation (input from simulation) | 0.565808 (s.e. 0.000046) | +0.00009 | `credit_sim.py`, `credit_sim_2026-11-23_154518.txt` |
| Neighbourhood model, no input from simulation of the tree | 0.56566 (s.e. 0.00001, Monte Carlo) | -0.00006 | `nb_closure.py`, `nb_closure_2026-11-24_150442.txt` |
| Cavity equations on degree paths, ratio equations (R), all seven runs combined | 0.565755 (s.e. 0.000012) | +0.000032 | `cav_run.py`, `cav_k5.py`, `cav_bias.py`; see below |
| Cavity equations on degree paths, Euler-Lotka equation (E), all seven runs combined | 0.565694 (s.e. 0.000042) | -0.000029 | as above |

The neighbourhood model follows a vertex with its children and its parent; the parent is followed from its own creation, with its own parent and all its children. Credits for states outside the followed region come from the followed parents by degree band, and the tail of the state distribution beyond degree 3000 is included. The model's mu is the root of the ratio equations with these credits, iterated to a fixed point. It is an approximation, not a bound: its distance to the reference, -0.00006 to -0.00007, is several Monte Carlo standard errors.

## Cavity equations on degree paths

The neighbourhood model stops at a fixed distance from the followed vertex. The cavity equations remove that cut. A vertex is described by its degree path, the times at which it gains children. Given that path, the credit it receives from each child and from its parent depends only on the subtree hanging from that neighbour, so the credit can be written as a message from the subtree. The equations ask for the distribution of these messages to reproduce itself when one more vertex is added.

The equations are solved by population dynamics, with no input from simulation of the tree:
- a population of records is kept for children (gains, credit, degree path) and for parents (the parent's path and its credit without the designated child);
- each generation builds new records from the previous generation's records; the change of path law is corrected by the exact likelihood ratio of the paths, and the records are resampled within each degree key;
- above degree 50 the path follows a large-vertex model: the difference between degree and non-leaf degree grows as a Yule process with drift 1 - 0.23 (0.23 is the credit per non-leaf neighbour of the neighbourhood model in the band 51-200);
- generation 0 starts from the neighbourhood model's tables (`nb_closure_2026-11-24_150442_tables.npz`).

The exponent is computed in two ways from the same records: (E) the root of the Euler-Lotka equation for the birth times of the children of a followed vertex, and (R) the root of the ratio equations with the credits measured on the followed vertices. There is no averaging over iterations: each value is the mean of independent replicas, with the standard error between replicas.

### Runs and checks

The decision rules were fixed before each run; the result files print them as PRE-REGISTERED lines.
- K2 (stability over generations). In `cav_run.py` each of generations 6-8 had to lie within 2 replica standard deviations of the replica means; this rule compares single generations with the spread of means, is stricter than the noise allows, and failed for both estimates. From `cav_k5.py` on it is replaced by K2': the slope of mu over generations 2-5 within 2 standard errors of zero. K2' passes for every variant.
- K3: smallest effective sample size of the reweighting at least 0.10. Passes in every run (smallest 0.23).
- K4: agreement with 0.565723 within 2 standard errors.
- K5 (`cav_k5.py`): the variants D (large-vertex model from degree 100, 2e5 records) and H (horizon 16 in age instead of 12) against the base B (degree 50, horizon 12, 3e5 records). The change of mu must lie within 2 combined standard errors. Passes for both estimates: the settings do not move mu at this precision.
- P1 (`cav_bias.py`): population size 3e4 and 1e5 against B (3e5), fit mu(N) = mu_inf + b/N. P2: drift 1 - 0.20 against 1 - 0.23 at 1e5.

| Run | records | (R) | (E) |
|---|---|---|---|
| `cav_run.py` | 3e5 | 0.565758 (0.000023) | 0.565619 (0.000157) |
| `cav_k5.py` B | 3e5 | 0.565782 (0.000021) | 0.565784 (0.000125) |
| `cav_k5.py` D | 2e5 | 0.565756 (0.000037) | 0.565709 (0.000061) |
| `cav_k5.py` H | 3e5 | 0.565739 (0.000029) | 0.565631 (0.000126) |
| `cav_bias.py` N30 | 3e4 | 0.565718 (0.000061) | 0.565821 (0.000269) |
| `cav_bias.py` N100 | 1e5 | 0.565716 (0.000038) | 0.565644 (0.000138) |
| `cav_bias.py` N100L (drift 1 - 0.20) | 1e5 | 0.565717 (0.000055) | 0.565665 (0.000120) |
| combined (weighted mean) | | 0.565755 (0.000012) | 0.565694 (0.000042) |

Standard errors in brackets. The seven (R) values are consistent with each other (chi-square 3.9 on 6 degrees of freedom), and so are the seven (E) values (chi-square 1.5).

### Status

(E) agrees with the reference value (difference -0.000029, 0.7 standard errors). (R) is higher by 0.000032, 2.7 standard errors. Against Uzay's certified interval, (E) is below its lower end by 0.5 standard errors and (R) above its upper end by 1.5 standard errors; the interval is narrower than these standard errors, so the cavity solution checks the equations and does not improve the value of mu. P1 finds no dependence on population size (bias at 3e5: -0.000009, s.e. 0.000007), and P2 finds no effect of the drift; the cause of the (R) difference is open. (E) and (R) differ by 0.000061 (s.e. 0.000044).

## Conclusions about the dynamics

A node gains children at a rate given by the credit it receives from its neighbours. The work above supports four conclusions; each is labelled as proved or observed in the numerics.

1. Proved, using the stable law of the tree seen from a non-leaf constructed by Uzay: the exponent depends on the tree only through the mean credit received by a newly created node in a given state and at a given age. The Euler-Lotka equation with the measured credit gives the exponent (row 2 of the table).
2. Observed: the exponent is set by young nodes. It responds to the credit from the parent in the first unit of age and to the credit received by nodes without leaf children in the first three units of age. Changing the credit of nodes of degree 50 and above has no effect at the precision of the runs; these nodes follow a Yule law with drift about 0.77.
3. Observed: equal sharing of credit misses the exponent by +0.003; following a node with its children, its parent and the parent's neighbours, with tables beyond, brings the difference to -0.00006; the cavity equations, which keep the whole tree, differ from that model by less than 0.0001. The two models also differ in their tables and settings, so this bounds the effect of correlations beyond two steps only together with those differences.
4. Proved: given the times at which a node and a neighbour gain children, the credit passed between them depends only on the subtree behind the neighbour, so the state of a node is its history of gains and not its degree. Observed: the credit table carries the selection along these histories (which nodes are still without leaf children at a given age), and no description by a few functions reproduces it. This is why no closed formula was found.

## Running

Python 3 with NumPy and Numba. Each script takes no arguments, has its settings fixed in the file, prints its progress and writes a new dated file to `code/results/`; nothing is overwritten.
- `closure_solver.py`: about one minute.
- `credit_sim.py`: simulation of the tree with credit tallies by state, 2 worker processes; a few minutes.
- `nb_closure.py`: about 27 minutes with the settings in the file; it reads `credit_sim_2026-11-23_154518_tallies.npz` for its comparisons with the measured credit, and writes its tables next to its result file.
- `cavity.py` is the solver module (imports `nb_closure.py` and `credit_sim.py`, reads `nb_closure_2026-11-24_150442_tables.npz`). The drivers, in this order: `cav_run.py` (about 16 minutes, one process), `cav_k5.py` (about 36 minutes; each variant runs in its own process with its own Numba cache), `cav_bias.py` (about 18 minutes; reads the B summary of `cav_k5.py`, `cav_k5_2026-11-24_233354_B.json`). Times on the author's machine with the settings in the files.
- `cr_exact.py`: exact distribution of the tree over unlabelled trees for N <= 16 in rational arithmetic, with the exact checks of the one-step relations (`cr_exact_2026-11-22_225355.csv`). `fresh_closure.py`: closure with new non-leaves treated exactly (`fresh_closure_2026-11-22_233326.txt`). These are the versions of 22 November 2026 that produced the result files.
- Sensitivity of the exponent to the credit (`red_s1.py`, `red_fwd.py`, `red_s3.py`, `red_s4.py`; `red_common.py` is read by `red_s1.py`). `red_s1.py` solves the cavity equations at a small size (4e4 records, four generations, a few minutes) and tallies the mean credit by state and age, split into the part from the parent and the part from the children (`red_s1_2026-11-24_210740_cloud.npz`). `red_fwd.py` solves the forward equation of one vertex with these credits by Crank-Nicolson and gives mu by Euler-Lotka; `red_s3.py` gives the change of mu when one part of the credit is raised by 1% in a block of states and ages; `red_s4.py` tests descriptions of the credit by a few functions. Each of the last three runs in under a minute and reads the tallies file.

## Earlier material

The 2022-2023 analysis in [`../CompleteRedirection2022/`](../CompleteRedirection2022/README.md) (Mathematica notebook, `CompleteRedirectionParts.pdf`, `Formulas.pdf`, Python notebooks) is the starting point of this work.

## License

Code under the repository's MIT license.
