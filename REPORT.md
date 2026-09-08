# Split inference on a heterogeneous ESP32 fleet: measuring the cost of the split point

**Status: results pending.** The system is built and verified; every table and
figure below has a placeholder that `analysis/figures.py` fills from the run
DBs. Run `README.md`, then `make figures`, then paste the tables from
`analysis/out/results.md` into the marked slots.

## 1. The question

A convolutional network can be cut in two: the first *k* layers run on the
microcontroller that holds the camera, the rest run on a server, and the
intermediate activation tensor travels over Wi-Fi between them. Where you put
that cut is a free parameter, and it trades three costs against each other:

- **compute on the device** - grows with *k*; the MCU is slow
- **bytes on the air** - depends on the shape of the tensor at the cut, and is
  *not* monotone in *k*
- **compute on the server** - shrinks with *k*; the server is fast

The two extremes are familiar. Cut at *k=0* and you have plain offload: send the
image, do nothing locally. Cut at the end and you have pure on-device inference,
which an ESP32 cannot do at a useful rate for this model. Everything in between
is *split inference*, and the interesting claim is that some intermediate cut
beats both endpoints - and that which cut wins depends on the hardware.

This report measures that claim on real hardware: three ESP32-class boards of
different capability, one Linux host as the server, a quantized MobileNetV2,
and four fixed splitting policies compared on identical inputs.

**What this report does not do:** it does not learn or adapt the split point.
Every policy here is a fixed rule. Adaptive orchestration is the natural next
question, and section 8 says where the groundwork for it lives.

## 2. Why the split point is not obvious

Intuition says "compute more on the device, send less". For a CNN with an early
stride-2 stage, that intuition is wrong in the region where it matters most.

A 96x96x3 int8 input is 27,648 bytes raw, but as a JPEG it is only ~1.8 KB -
the image codec is extremely good at images. The activation tensor after the
first block, by contrast, is *uncompressed* and has more channels than the
input. Splitting early therefore **inflates** what goes on the air relative to
simply sending the JPEG. Only once the network has downsampled several times
does the activation get smaller than the compressed image.

That is why full offload (`k0`) is in the comparison as a policy, not merely as
a strawman: a split point has to *earn* its place by beating the JPEG.

The measured version of this argument is Table 1.

## 3. System

### 3.1 Hardware

| Node | Tier | Board | Memory | Role |
|---|---|---|---|---|
| 11 | A | ESP32-S3 devkit | PSRAM, vector ISA | strongest device tier |
| 21 | B | ESP32 + PSRAM | PSRAM, no vector ISA | mid device tier |
| 31 | C | ESP32, plain | ~150-200 KB internal SRAM | weakest device tier |
| - | - | x86 Linux laptop (Core 7 150U) | - | soft-AP, server tail, orchestrator, logging |

Three tiers exist so that "the best split point depends on the device" is
testable rather than asserted. The host runs the server tail **on its CPU**,
with the orchestrator pinned to four cores (`taskset -c 0-3`) so that other work
on the machine cannot leak into the `queue` column.

That pinning does **not** make this an edge-class server, and the report does
not pretend otherwise. Measured with `make bench-tail` on four pinned cores, the
tail costs **0.29 ms for `k0`** (whole model + JPEG decode), **0.13 ms for
`k_shallow`** and **0.07 ms for `k_deep`** - one to two orders of magnitude
below what the same work costs on a Pi-class CPU, and unchanged by the pinning
(0.24 ms unpinned for `k0`). The `server` stage is therefore effectively zero
here, and so is any queueing behind it. section 7 states what that does to the
conclusions; it is the single most important caveat in this report, and section 5.7
projects what a slower server would have changed.

The frequency ceiling is part of that definition and is capped for the session
(`README.md`); `make bench-tail` prints the ceiling it measured
under, and that number belongs in this section alongside the tail costs.

The host is also the Wi-Fi access point, so the channel is ours: no other
traffic, a fixed channel, and the ability to inject loss with `netem` for the
correctness checks. The fleet AP is 2.4 GHz because ESP32 radios have no 5 GHz
band.

### 3.2 Model and split points

MobileNetV2, width multiplier 0.35, 96x96x3 input, full-int8 post-training
quantization, two classes (Visual Wake Words: person / no-person).

The heads and tails are **sliced from one set of weights at the flatbuffer
level**, with the boundary tensor's quantization parameters (scale, zero-point)
copied verbatim from head output to tail input. Cuts are taken only at
single-tensor boundaries - the output of an inverted-residual block, never
inside one, so no residual connection is severed.

The consequence is the property the whole comparison rests on:

> For any cut *k* and any input *x*: `tail_k(head_k(x)) == full(x)`, bit-exact on
> the int8 logits.

This is enforced as a test (`make test-slow`, 200 images, every cut) and it is
gate **G0** in the measurement plan. Because accuracy is *identical by
construction* across split points, accuracy drops out of the comparison entirely
and latency is the only thing being measured. That is a deliberate
simplification and it is why this study can be about latency alone.

| Cut | Boundary | Activation shape | Activation bytes (int8) |
|---|---|---|---|
| `k0` | none - full offload | JPEG image | ~1,775 (dev assets; real VWW ~ 3-5 KB) |
| `k_shallow` | after inverted-residual block 2 | 24x24x8 | 4,608 |
| `k_deep` | after inverted-residual block 6 | 6x6x24 | 864 |

Head cost on flash: `k_shallow` 17,120 B / 12 ops, `k_deep` 59,952 B / 26 ops.

**Feasibility is per tier.** A head is only an option on a board whose TFLM
arena fits it. Tier C has no PSRAM, so `k_deep` is expected to be infeasible
there and possibly `k_shallow` too. This is checked on device at boot (gate
**G3**) rather than assumed, and the answer is itself a result - see section 5.6.

### 3.3 Input path - the controlled comparison

Every node stores the **same 50 test images** in a SPIFFS partition, in two
representations generated by one script from the same sources: the JPEG (what
goes on the air for `k0`) and the pre-processed raw int8 96x96x3 tensor (what
feeds the head for every deeper cut). The image used for request *n* is
`n mod 50` on every node.

So every policy, on every node, sees identical inputs in identical order. No
live camera is involved: capture jitter would add variance to precisely the
quantity being compared, and only one board has a camera anyway.

### 3.4 Transport

Custom protocol over UDP, little-endian packed structs, no JSON on device.
Two ports: control (heartbeat, assignment, ack, abort, result) and data (tensor
fragments, NACKs).

- Tensors are fragmented at 1,400 B payload - one 802.11 frame, no IP
  fragmentation.
- The last fragment carries a 12-byte trailer: CRC32 of the tensor plus the
  device-measured `t_capture_us` and `t_edge_us`.
- Reliability is **NACK-based selective repeat**: 20 ms after the last fragment,
  if the tensor is incomplete, the server sends a bitmap of what is missing and
  the device resends only those fragments. Maximum 3 rounds, then the request is
  aborted. The common case is 0-2 losses per tensor, so NACK costs far less than
  per-fragment ACK, and the 3-round bound keeps the worst case finite - which is
  what makes a p95 comparable across policies.

This path is verified under *injected* loss rather than ambient luck: 200
tensors at each of {0, 2, 5, 10}% loss with CRC verification, plus a 40% run to
confirm the abort path fires (gate **G1**).

### 3.5 Orchestration and timing

One asyncio process on the host. Per request: pick the cut for this node -> send
`ASSIGN` (acked within 100 ms, 3 retries, then the node is marked lost) -> wait
for the reassembled tensor (bounded by `t_max_ms`) -> run the server tail -> send
the result back -> write one log row.

At most one request in flight per node, and **no fleet-wide cap**. An earlier
version capped fleet concurrency at two; on a three-node fleet that made
nodes block each other from dispatching, so a node's measured latency
depended on which cut the *other* nodes had been assigned - a confound in
precisely the quantity being compared (section 7). Nodes now interact only through
the shared radio and the server tail's queue, and queue time is one of the
reported stages, so whatever coupling remains is visible in the data rather
than hidden in it.

**All reported times are durations, never cross-device timestamps.** The device
measures its own capture and inference with its local monotonic clock and ships
those two numbers in the trailer; everything else is measured on the server's
monotonic clock. No clock synchronisation is needed, and none is performed.

One request row decomposes into five non-overlapping stages that sum to the
end-to-end latency:

| Stage | Definition |
|---|---|
| `device` | `t_capture_us + t_edge_us` - SPIFFS read + head inference on the MCU |
| `uplink` | dispatch to last fragment, minus device time: `ASSIGN` RTT + airtime + retransmissions |
| `queue` | waiting for the server tail's single worker |
| `server` | tail inference on the host (includes JPEG decode for `k0`) |
| `residual` | result dispatch and scheduler overhead |

## 4. Method

### 4.1 The policies compared

| Policy | Rule |
|---|---|
| `k0` | never split - JPEG to the server, full model on the server |
| `k_shallow` | always split after block 2 |
| `k_deep` | always split after block 6, clamped per tier to the deepest cut that tier can run |
| `best` | per node, the cut with the lowest mean latency in the sweep |

`best` is not a fifth strategy but a consistency check: it is by construction the
argmin over the sweep, so replaying it should reproduce the winning fixed
policy's numbers on each node. A gap means conditions drifted between the two
sessions, and the runs should be repeated (gate **G7**).

### 4.2 The two runs

1. **Sweep - isolated.** Every feasible `(node, cut)` pair, 200 requests each,
   in blocks, **one node at a time**: node 11 runs all its cuts while 21 and 31
   sit idle, then node 21, then node 31. This is the primary dataset - the
   per-cut latency distributions, the stage breakdown, and `best_table.json`.

   The isolation is not fastidiousness. It was added after a concurrent sweep
   gave the wrong answer: the three nodes share one single-worker server tail,
   so a block that happened to overlap a cheap-tail block on another node
   measured faster than a block overlapping an expensive-tail one. In a
   simulated run that flipped node 11's argmin - the sweep picked `k_deep`
   (71.2 ms) over `k_shallow` (75.5 ms), while the isolated head-to-head runs
   had `k_shallow` winning 66.6 ms to 70.3 ms. Measuring one node at a time
   removes the confound.

2. **Policy runs - fleet-concurrent.** One run per policy, 200 OK requests per
   node, all nodes active. This is the head-to-head comparison and the source
   of the CDFs.

   These are deliberately concurrent: deploying a policy means every node runs
   it, so a fleet-level comparison is the operationally meaningful one. The
   price is that a node's number here depends on what the rest of the fleet is
   doing, so **Table 4's per-node values are not directly comparable to Table
   2's**. Comparing the same `(node, cut)` across the two is itself informative:
   the difference is the cost of sharing the server.

Both are executed back to back in a single session with fixed node placement, so
the split point is the only thing that varies.

### 4.3 What is measured

Per request: end-to-end latency and all five stage components, bytes on the air,
fragment count, retransmission count, RSSI at dispatch, final status
(`OK` / `TIMEOUT` / `CRC` / `LOST`). Everything lands in one SQLite row; every
number below is a query over those rows and nothing is transcribed by hand.

### 4.4 What is *not* controlled

Single site, single channel, one model family, one session. Ambient 802.11
activity from outside the lab is not controlled - only the absence of *our own*
traffic is. See section 7.

### 4.5 Gates

Each step of the measurement is gated; a gate that fails invalidates the data
below it rather than being worked around.

| Gate | Protects |
|---|---|
| G0 identity | that every policy runs the same model, so latency is the only difference |
| G1 ARQ integrity | that a delivered tensor is the tensor that was sent |
| G2 harness | that a hardware surprise is about hardware, not about the code |
| G3 arena | that an assigned cut is one the device can actually execute |
| G4 clean fleet | that latency rows are not contaminated by reboots or lost nodes |
| G5 loss | that the radio path does not corrupt under realistic loss |
| G6 samples | that p95 means something |
| G7 dispatch | that `best` actually ran the cut the sweep selected |

Isolation is gated too: the sweep measures one node at a time, so a per-cut
latency is a property of the cut and not of what the other nodes were doing.

## 5. Results

### 5.1 What each split point puts on the air

<!-- PASTE: analysis/out/results.md section T1 -->

| cut | activation bytes | mean bytes on air | mean fragments | retransmits / request |
|---|---|---|---|---|
| k0 | 1,775 (JPEG) | - | - | - |
| k_shallow | 4,608 | - | - | - |
| k_deep | 864 | - | - | - |

*Read this first.* If `k_shallow` sends more bytes than `k0`, the data-inflation
effect of section 2 is confirmed on real traffic - and any latency win `k_shallow`
shows must come from moving work off the server, not from sending less.

### 5.2 Latency per node and split point

![Latency per split point](analysis/out/latency_by_cut.png)

<!-- PASTE: analysis/out/results.md section T2 -->

| node | cut | n OK | mean ms | p50 ms | p95 ms | failed |
|---|---|---|---|---|---|---|
| 11 (A) | k0 | - | - | - | - | - |
| 11 (A) | k_shallow | - | - | - | - | - |
| 11 (A) | k_deep | - | - | - | - | - |
| 21 (B) | k0 | - | - | - | - | - |
| 21 (B) | k_shallow | - | - | - | - | - |
| 21 (B) | k_deep | - | - | - | - | - |
| 31 (C) | k0 | - | - | - | - | - |
| 31 (C) | k_shallow | - | - | - | - | - |

Measured with one node active at a time, so these are properties of the
`(node, cut)` pair alone.

**The question this table answers:** does the winning cut differ between tiers?
If tier A's argmin is a deep cut while tier C's is `k0`, the headline result is
that *the right split point is a property of the device, not of the model* -
which is exactly why a single static split for a whole fleet is the wrong
design. If instead one cut wins everywhere, that is an equally publishable
negative result and section 6 should say so plainly.

### 5.3 Where the time actually goes

![Stage breakdown](analysis/out/stage_breakdown.png)

<!-- PASTE: analysis/out/results.md section T3 -->

| node | cut | device | uplink | queue | server | residual |
|---|---|---|---|---|---|---|
| 11 (A) | k0 | - | - | - | - | - |
| ... | ... | | | | | |

This is the diagnostic table: it says *why* a cut wins or loses, not just that
it does. Expected shape, to be confirmed or refuted:

- `k0` should be dominated by `server` (the host runs the whole model,
  including JPEG decode) with a small `device` term.
- `k_shallow` should show the largest `uplink` (most bytes, most fragments) and
  a modest `device` term.
- `k_deep` should show the largest `device` term and the smallest `uplink` and
  `server` terms.
- `queue` is the cross-node coupling term: it is the only stage where what
  the *other* nodes are doing shows up. If it is small, the per-node numbers
  can be read independently. If it is large - particularly under `k0`, whose
  tail runs the whole model plus a JPEG decode - then the fleet is
  server-bound and the policy comparison is partly a comparison of server
  load, which must be said out loud in section 6.

### 5.4 Policy comparison

![Latency CDFs per policy](analysis/out/policy_cdfs.png)

<!-- PASTE: analysis/out/results.md section T4 -->

| policy | node | cut actually run | n OK | mean ms | p95 ms |
|---|---|---|---|---|---|
| k0 (no split) | 11 (A) | k0 | - | - | - |
| ... | | | | | |

Measured with the whole fleet active, so each column reflects that policy's
fleet composition. Compare policies to each other here; compare split points
in isolation in Table 2. The gap between a node's number here and its Table-2
number for the same cut is the cost of sharing the server tail.

The CDFs matter more than the means. Two policies with the same mean can have
very different tails, and for a fleet doing repeated inference the tail is what
determines the achievable rate. Look specifically at whether the winning policy
also wins at p95, or whether it trades a better mean for a worse tail.

### 5.5 The cost of running the fleet concurrently

<!-- PASTE: analysis/out/results.md section T5 -->

| policy | node | cut | isolated (T2) | concurrent (T4) | difference |
|---|---|---|---|---|---|
| k0 (no split) | 11 (A) | k0 | - | - | - |
| k_shallow | 11 (A) | k_shallow | - | - | - |
| ... | | | | | |

The same node running the same cut, measured alone in the sweep versus with the
whole fleet active. The difference is what sharing one server tail costs.

This is the column to watch for the argument the whole project rests on. A
policy whose server work is expensive - `k0` runs the entire model plus a JPEG
decode on the host - does not only pay for itself, it makes every other node
wait.
If that shows up here, then **the case for splitting is stronger at fleet scale
than the isolated numbers of Table 2 suggest**, because splitting moves work off
the one resource every node is queueing for.

An earlier simulation showed exactly that, and large: `k0` cost +13% to +44%
per node once the fleet ran concurrently, against -0.4% to +2.7% for the split
cuts. **That simulation assumed a Pi-class server tail.** With the actual
server measured at 0.07-0.29 ms per request (section 3.1), the same simulation shows
the asymmetry gone - every policy within +/-9% - because there is no longer a
queue to contend for. This table is expected to be flat, and if it is, that is
a statement about this server, not about split inference.

### 5.6 Feasibility: which split points each tier can host

| Tier | Board | `k_shallow` arena | `k_deep` arena | Feasible cuts |
|---|---|---|---|---|
| A | ESP32-S3 (PSRAM) | - B | - B | - |
| B | ESP32 (PSRAM) | - B | - B | - |
| C | ESP32 (no PSRAM) | - B | - B | - |

Filled from each board's boot log (`ARENA cut=... used=...`). If tier C turns out to
support only `k0`, that is a finding: on the weakest hardware the orchestrator's
only rational choice is not to split at all, and the heterogeneity of the fleet
lives in tiers A and B.

### 5.7 What a slower server would change

![projected latency vs server speed](analysis/out/server_sensitivity.png)

<!-- PASTE: analysis/out/results.md section T6 -->

| node | best cut on this server | cut that takes over | server slower by | k0 tail cost at crossover (ms) |
|---|---|---|---|---|
| 11 (A) | - | - | - | - |
| ... | | | | |

The server here is fast enough that `server` and `queue` barely register (section 3.1),
which favours full offload. This section asks what would change on a slower one
**without pretending to have measured it**: hold the measured `device`, `uplink`
and `residual` stages fixed - they are properties of the boards and the radio,
and a different server does not touch them - and scale the measured `server`
stage by a factor. The crossover is where a split cut overtakes `k0`.

Two things make this a projection rather than a result, and both are stated on
the figure's own terms:

- The server axis is assumed, not measured. Only its *shape* is
  measured - the per-cut ratio of tail costs is real, since all three tails were
  timed on the same machine with the same interpreter.
- Queueing is excluded, because a single-worker trace cannot predict it. A
  slower server also queues, and queueing costs whichever policy leans on the
  tail hardest - which is `k0`. **The real crossover therefore arrives earlier
  than this table says**, making every number here conservative.

What it is good for: reading the report's headline result off a different
deployment. If your edge server does the full model in, say, 40 ms, this is the
table that says whether splitting would have paid.

### 5.8 Reliability

| Run | requests | OK | TIMEOUT | CRC | LOST | reboots |
|---|---|---|---|---|---|---|
| sweep | - | - | - | - | - | - |
| k0 | - | - | - | - | - | - |
| k_shallow | - | - | - | - | - | - |
| k_deep | - | - | - | - | - | - |
| best | - | - | - | - | - | - |

`CRC` must be zero. A non-zero `reboots` column for a node invalidates that
node's rows for that run (see the power note in `README.md`) and the
run must be repeated.

## 6. Discussion

> **To write once section 5 is filled.** Answer, in order:
>
> 1. Does an intermediate split beat full offload at all, and on which tiers?
> 2. Does the winning split point differ across tiers? By how much latency?
> 3. Does the stage breakdown explain the winner, or is something unaccounted
>    for (a large `residual` or `queue` would mean the model of section 3.5 is wrong)?
> 4. Is the data-inflation effect visible in Table 1, and does it match the
>    activation sizes predicted from the model architecture?
> 5. Does the mean-optimal policy also win at p95?
> 6. How much does Table 5 change the ranking? If concurrency punishes `k0`
>    disproportionately, the fleet-scale case for splitting is stronger than the
>    isolated numbers show, and that belongs in the conclusion.
> 7. If one policy wins everywhere: say so directly. "A single static split is
>    sufficient for this model on this fleet" is a real answer, and the sweep is
>    the evidence for it.

## 7. Threats to validity

- **The server is far faster than the edge server this study was designed
  around, and this decides part of the result.** The design called for a
  Raspberry Pi 4; that board is no longer available, so the x86 workstation that
  builds the firmware also runs the AP, the orchestrator and the tail. Measured:
  0.29 / 0.13 / 0.07 ms per request for `k0` / `k_shallow` / `k_deep`
  (`make bench-tail`, four pinned cores). Two consequences, both structural:
  1. **The `server` and `queue` stages are effectively zero**, so the
     five-stage decomposition collapses to `device` + `uplink`. The cost of
     sharing a server tail - Table 5, and the strongest argument for splitting
     at fleet scale - cannot be observed on this hardware. Its cells are
     expected to be ~0%, and that is a property of the server, not a refutation.
  2. **The comparison is biased toward `k0`.** Full offload is the policy that
     puts the most work on the server, so a near-free server is worth most to
     it. A win for `k0` here is therefore weak evidence; a win for a *split* cut
     would be strong evidence, since it would have to come entirely from
     `device` + `uplink`.

  What survives unaffected: Table 1 (bytes on air), the `device` and `uplink`
  columns of Table 3, and the arena feasibility table (section 5.6). Those are
  properties of the boards and the radio, and they are what this report can
  claim. Anything that depends on the server's speed is reported as measured
  and read with this bullet in view - and section 5.7 projects the measured stages
  onto a slower server, which is the closest this hardware can get to the
  question.

  Two mitigations were tried. Capping the CPU frequency (400 MHz floor via
  `intel_pstate`) is a real, smooth slowdown and is used for the session. A
  cgroup CPU quota was rejected after measurement: at a 1 ms period - the
  kernel's floor - `CPUQuota=10%` puts the mean `k0` tail at 13.8 ms, which
  looks right, but min stays at 0.26 ms and p95 rises to 30 ms. That is a fast
  CPU stalling, not a slow CPU, and the artificial tail would land hardest on
  the policy that uses the server most, which is the effect under test.
- **Nothing else may run on the pinned cores.** A compile or a browser
  competing for cores 0-3 lands in `server` and `queue`. Sessions where that
  happened must be discarded, not corrected.
- **Single session, single site.** RSSI and ambient 802.11 occupancy change
  between days. All runs are back to back with fixed placement for exactly this
  reason, but the numbers are not portable to another room.
- **No repeats, no confidence intervals.** One run per policy. Differences
  smaller than the run-to-run spread should not be read as real; the p50/p95
  columns and the CDFs are more trustworthy than a mean difference of a few
  percent.
- **The model is a placeholder unless retrained.** The dev pipeline ships an
  ImageNet-initialized MobileNetV2-0.35 with an untrained 2-class head. Its
  *architecture, quantization, and tensor shapes* are those of the real model, so
  all latency and byte measurements are valid - but its accuracy is meaningless.
  Real Visual Wake Words training (`model/train.py`, >=0.80 int8 accuracy gate)
  runs on the workstation and changes no timing.
- **Stored images, not live capture.** Capture cost on a real camera is not
  included in `device`. This makes the comparison cleaner and understates
  absolute latency for a camera deployment by the capture time.
- **JPEG size depends on image content.** `k0`'s bytes on air vary per image
  (dev assets: 1,362-2,326 B); the other cuts are fixed-size. Table 1 reports the
  mean, and all policies see the same 50 images, so the comparison is fair - but
  `k0`'s latency has a variance source the others lack.
- **Cross-node coupling through the server tail is real and not eliminated.**
  The nodes share one single-worker tail, so a policy that gives the other
  nodes expensive tail work - notably `k0`, whose tail decodes a JPEG and runs
  the whole model on the host - raises this node's queue time too. This is not
  hypothetical: during harness validation a concurrent sweep flipped a node's
  argmin (section 4.2), which is why the sweep now measures one node at a time. The
  policy runs remain concurrent by design, so Table 4 carries the coupling and
  Table 3's `queue` column measures it. Read any cross-policy difference on a
  single node with that column in view.
- **Three nodes with one request each in flight is not radio contention.**
  The study says nothing about what happens when many devices transmit
  simultaneously - see section 8.

## 8. What was deliberately left out

This project was scoped down to one question: *what does the split point cost,
and does the answer depend on the device?* The following were cut to keep it
answerable:

- **Adaptive / learned split selection.** A contextual bandit that picks the cut
  per device per request. The orchestrator's policy interface (`select(node) ->
  cut`) is the seam a learner plugs into, and the sweep in section 5.2 is exactly the
  ground truth such a learner would have to beat.
- **Radio contention at fleet scale.** What happens when six devices transmit
  simultaneously through one access point. Three nodes, each with a single
  request in flight, do not generate it.
- **Induced dynamics.** Scripted CPU throttling and background traffic, to make
  the environment non-stationary. Without a learner there is nothing for
  non-stationarity to demonstrate.
- **Energy per inference**, live camera capture, transport alternatives
  (TCP, ESP-NOW), and per-packet encryption.

The code for the learner, the dynamics, and the clock sync is preserved on
the local `full-system` git branch.

## 9. Reproducing this

```bash
make env                                # python 3.11 venv via uv
make model-dev slice assets-dev         # model, head/tail slices, flash assets
make test test-slow                     # unit tests + bit-exact split identity gate
make preflight                          # whole sequence against simulated nodes
make header fw-all                      # firmware, one project per board
scripts/flash_all.sh esp32s3=... esp32cam=... esp32=...
# host AP and the measurement sequence: README.md
make figures                            # every figure and table in section 5
```

Raw evidence: the SQLite DBs in `runs/`. Each stores the complete `config.yaml`
and the orchestrator's git hash alongside its rows, so any figure traces back to
the exact configuration that produced it.
