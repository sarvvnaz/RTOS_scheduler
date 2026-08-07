# Assumptions

Everything in the simulation that was a **choice** rather than something
the project definition or a paper stated outright. Each one is somewhere a
result could change if the choice were made differently, so each is worth
checking before the numbers are quoted.

Marked **[SPEC]** where the definition says it explicitly, **[PAPER]**
where a cited paper does, and **[MINE]** where neither did and I picked.

---

## 1. Task generation

**[SPEC] Communication cost per edge is `U(0.5·Avg, 1.5·Avg)` where
`Avg = CCR × C_i`, and `C_i` is the whole task's execution time.**
This is written exactly as the definition gives it, but the consequence is
severe: `Avg` grows with the number of nodes while the benefit of moving
one node does not. Offloading breaks even only when `CCR < 1/(2·|V|)`,
about 0.045 for an 11-node task and under 0.025 for the 20–50 node graphs
the definition asks for. The definition's CCR range starts at 0.25.
**This is the single most consequential open question in the project.**

**[MINE] Communication is paid only when the two nodes are on different
servers.** The definition has two sentences: one says the cost applies
when nodes are "on two different processors of two different servers", the
other says two nodes on the same processor pay nothing. Neither covers two
*different* cores of the *same* machine. I read the first sentence as the
operative rule, so two local cores talk free. Under the stricter reading
the mapper collapses every node onto one core, which is what produced the
first, broken set of charts.

**[MINE] A task whose critical path exceeds its deadline is redrawn.**
Such a task cannot be scheduled on any number of cores, so counting it as
a scheduling failure blames the analysis for the input. Up to 20 attempts,
then the setting is reported as a generation failure. Without this, 4.6%
of tasks at `u_norm=0.5` were impossible from the start.

**[MINE] Per-task utilization is bounded only by `0.9 × min(node count)`.**
This lets one task hold about half of the system's load, which looks
extreme but is exactly what uniform sampling over the simplex gives
(observed max shares 0.519 / 0.427 / 0.350 against the theoretical
`H_n/n` of 0.521 / 0.408 / 0.340). Left as RandFixedSum produces it,
because capping it would bias the distribution away from the standard
method. The definition also explicitly allows `u_i > 1`.

**[SPEC] Periods drawn from {2000, 4000, 6000}; deadlines implicit
(`D = T`); Erdos-Renyi with p = 0.1.** Edges only run from a lower node id
to a higher one, which guarantees acyclicity — a generation convenience,
not something the definition requires.

**[SPEC] CSP is the share of a node's execution spent in critical
sections**, split among its sections with RandFixedSum, and accesses are
non-nested because segments run one after another.

**[USER DECISION] No mixed criticality.** The definition describes a
two-level mixed-criticality system with normal and overrun modes. It was
implemented and then removed on instruction. Every node has one execution
time.

---

## 2. Mapping (OC-HEFT)

**[MINE] Cost weights `w1 = w2 = w3 = 1`.** The definition gives
`Cost = w1·Exec + w2·Comm + w3·RC` but no values, and these decide the
whole experiment: at `w2 = 1` the communication term dominates and the
mapper co-locates aggressively; at `w2 = 0` it spreads over 17 cores.
The three terms are on very different scales, so equal weights are not a
neutral choice.

**[MINE] The per-core available time resets between tasks.** Tasks are
periodic and each is judged against its own deadline, so a node must fit
inside its own release rather than queue behind another task on one global
timeline. Capacity sharing across tasks is left to the utilization
constraint. Without this every task after the first was pushed past its
deadline.

**[MINE] Communication is counted only from already-placed
predecessors.** Successors have no core yet when a node is placed, so
their edges cannot be priced. This makes the cost order-dependent.

**[MINE] `Usage_j(r)` is measured over one hyperperiod**, matching the
definition's "R1 = 60%" figure but restricted to a single core.

**[MINE] Source and sink are exempt from both constraints.** They take
zero time, so they can never violate a utilization or deadline bound.

---

## 3. Federated scheduling

**[PAPER] A task is heavy when `u_i > 1`, and gets
`m_i = ceil((C_i - L_i)/(D_i - L_i))` dedicated cores.** Light tasks share
whatever is left.

**[PAPER] Cluster sizes grow until every heavy task meets its deadline**,
failing when the heavy tasks together want more cores than exist. This is
Algorithm 1. **[MINE]** the loop is capped at 12 rounds; a set that has
not settled by then is reported unsettled rather than assumed schedulable.

**[MINE] `m_i` is computed at the speed of the cores the cluster will
receive**, since a faster cluster genuinely needs fewer cores.

**[MINE] Faster cores are handed out first, and the largest task is served
first.** This favours putting heavy tasks on edge servers. A different
order would give different clusters and different results.

**[MINE] A cluster is taken from a single speed class where possible**, so
clusters are uniform and `cluster_speed` is well defined.

**[MINE] A light task's `m_i` is the number of cores its nodes actually
used**, not the size of the pool it had access to. Dividing by the whole
pool would credit it with cores it never touched.

---

## 4. Protocols

### MSRP

**[STANDARD] FIFO spin: a request waits behind at most one request from
each other core using the resource.** Its own core cannot compete with
itself.

**[MINE] Arrival blocking is charged once per job per core the task uses**,
not once per node. Charging it per node made a core hosting 49 nodes pay
it 49 times and pushed utilization from 0.9 to 1.54 on its own.

### POMIP

**[PAPER] `F^I` and `F^O` follow Lemmas 8 and 10; the response time is
equation (11).**

**[MINE, PROVEN] Only `N^lambda = 0` and `1` are checked** instead of the
full sweep to `N_iq`. `F^I` decreases in that count and `F^O` is constant
above 1, so no other value can win. Verified identical to the full sweep
over 180 task-resource pairs.

**[MINE] `C_i` and `L_i` are divided by the cluster speed; the blocking
term is not.** Blocking is caused by other tasks on their own cores, so
this task's speed should not shorten it. Mixing scaled and unscaled terms
in one bound is defensible but worth challenging.

**[MINE] Context switches cost `2 × context_switch` per request**, and the
value is 1.0. That works out at 0.22% of total blocking, so the
definition's requirement to account for context-switch overhead is
satisfied in form more than in effect. There is no stated value to use,
and the answer depends on what one time unit is meant to represent.

### H2LP

**[PAPER] The response time follows Corollary 4.9,
`R_i <= L_i + B^S + B^T + B^A + I_i/n_i`**, with token blocking for heavy
tasks and arrival blocking for light ones.

**[MINE] This is the protocol's mechanism, not the paper's analysis.** The
paper bounds blocking with a linear program that avoids counting any
request twice. Mine is a direct, simpler bound and is therefore more
pessimistic. **H2LP results here should not be read as reproducing the
paper's numbers.**

**[MINE] Token blocking is `(N_iq - 1) × L_iq` for a heavy task** — the
task's own other requests, each held for its longest critical section.

**[PAPER] A light task's cluster is a single core.** The paper schedules
light tasks as sequential tasks, one processor each, so H2LP reduces to
MSRP for them. Treating the whole leftover pool as one cluster instead
drove the spin bound to exactly zero and made H2LP look unbeatable.

### All three

**[MINE] Every protocol is judged by the same response-time test.**
Equation (11)'s shape is reused for MSRP and H2LP with each protocol's own
blocking term. Applying it to POMIP alone, as an earlier version did,
failed POMIP for a reason the others were never asked about — 7 of 60 sets
failed on that test alone.

**[NOT IMPLEMENTED] The adaptive protocol.** Its slot is reserved and it
raises rather than silently doing nothing.

---

## 5. Schedulability test

**[SPEC] Partitioned EDF, per-core queues, no migration after mapping.**

**[MINE] A core passes if `sum (Exec + on-core blocking) / T <= 1`.**
Which blocking counts as on-core is the whole protocol difference: spin
and context switches occupy the core, suspension and token waiting do not.

**[MINE] A task counts as missed if any of its nodes went unmapped, if it
shares an overloaded core, or if it fails the response-time bound.** This
per-task verdict is what makes quality of service meaningful; before it,
QoS only counted POMIP misses and sat pinned at 1.00 for MSRP.

---

## 6. Experiments

**[SPEC] 100 task sets per point.** Fewer only where generation failed —
89 of 100 at the highest load, and those are reported.

**[MINE] Each chart has its own base configuration.** One setting cannot
put all eight parameters in the range where they matter: light enough for
the load sweep to show a curve leaves resource count doing nothing, and
heavy enough for resource count to bite leaves every load point already
failing. Each chart prints its conditions, and they are in the csv.

**[MINE] The two edge charts use `CCR = 0.05`, not the definition's
0.25.** At 0.25 only 1.4% of nodes are offloaded and the charts are flat;
at 0.05 it is 47–55% and the platform is visible. **This is an assumption
that changes what those two charts say, and it exists only because of the
communication-cost question in section 1.**

**[MINE] The clustering charts run on m=8 with no edge servers**, as the
H2LP paper does. With the edge platform attached there are 24 cores for at
most 8 tasks, so exclusive clustering never runs short and the question
cannot be observed.

**[MINE] CSP is sampled at 0.1, 0.25, 0.5, 0.75, 1.0.** The definition
gives a range, not a set.

**[SPEC, CONFOUNDED] Three sweeps move a second variable with the one they
name**, and all three follow from the definition's own formulas, so they
are labelled rather than silently corrected:

- core count: `U = m × U_norm`, so load rises 0.6 to 4.8 across the sweep
- resource count: total requests are `n_r × requests-per-resource`, so
  locking volume rises with the number of types
- task count: `U` and total requests are both fixed, so more tasks means
  smaller tasks that lock less often

**[MINE] A setting the generator cannot build is skipped, not counted as
unschedulable.** That is a limit of the generator, not of the scheduling,
and the count is reported separately.

---

## The three worth resolving first

1. **Is `Avg = CCR × C_i` per edge intended?** It decides whether the edge
   platform is usable at all, and therefore what half the charts mean.
2. **What are `w1, w2, w3`?** They decide whether the mapper spreads or
   co-locates, which changes every downstream number.
3. **Should MSRP have a response-time test, or only the utilization one?**
   It decides whether the protocol comparison is symmetric.
