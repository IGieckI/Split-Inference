# Split inference on a heterogeneous ESP32 fleet: measuring the cost of the split point

**Status: complete.** Every table and figure below is generated from the run
DBs by `analysis/figures.py` - regenerate them with `make figures`. The
procedure that produced the runs is in `README.md`.

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

The **CPU is capped to 400 MHz** for the session (`intel_pstate`'s floor;
`README.md`), which is the other half of the server's definition.
Measured with `make bench-tail` on four pinned cores at that cap:

| cut | mean | p95 | min |
|---|---|---|---|
| `k0` (whole model + JPEG decode) | 3.45 ms | 3.51 ms | 3.38 ms |
| `k_shallow` | 1.36 ms | 1.38 ms | 1.34 ms |
| `k_deep` | 0.76 ms | 0.78 ms | 0.71 ms |

Uncapped, the same three cost 0.29 / 0.13 / 0.07 ms, so the cap buys a ~12x
slowdown. Two properties make it usable as a measurement rather than a
distortion: it is a genuinely slower processor, not an intermittently
unavailable one - min 3.38 ms against p95 3.51 ms, a distribution as tight as
the uncapped one - and it is the same slowdown for every policy, so no cut is
advantaged by it.

This still is not a Pi-class server; it is perhaps a third to a seventh of the
way there. section 7 states what that costs the conclusions, and section 5.7 projects the
measured stages onto a slower server.

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
arena fits it. This is checked on device at boot (gate **G3**) rather than
assumed, and the answer is itself a result: tier C hosts neither cut and its
firmware links no head at all - see section 5.6.

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

<!-- T1 - generated by analysis/figures.py; regenerate with `make figures` -->

| cut | activation bytes | mean bytes on air | mean fragments | retransmits / request |
|---|---|---|---|---|
| k0 | 1775 | 1811 | 1.98 | 0.000 |
| k_shallow | 4608 | 4668 | 4.00 | 0.000 |
| k_deep | 864 | 888 | 1.00 | 0.000 |

*Read this first.* If `k_shallow` sends more bytes than `k0`, the data-inflation
effect of section 2 is confirmed on real traffic - and any latency win `k_shallow`
shows must come from moving work off the server, not from sending less.

### 5.2 Latency per node and split point

![Latency per split point](analysis/out/latency_by_cut.png)

<!-- T2 - generated by analysis/figures.py; regenerate with `make figures` -->

| node | cut | n OK | mean ms | p50 ms | p95 ms | failed |
|---|---|---|---|---|---|---|
| 11 (A) | k0 | 200 | 32.6 | 29.8 | 55.5 | 0.0% |
| 11 (A) | k_deep | 200 | 190.0 | 189.9 | 209.6 | 0.0% |
| 11 (A) | k_shallow | 200 | 166.8 | 167.5 | 184.2 | 0.0% |
| 21 (B) | k0 | 200 | 36.0 | 33.0 | 63.8 | 0.0% |
| 21 (B) | k_deep | 200 | 719.9 | 720.3 | 743.6 | 0.0% |
| 21 (B) | k_shallow | 200 | 557.8 | 557.7 | 581.4 | 0.0% |
| 31 (C) | k0 | 200 | 41.5 | 40.4 | 68.4 | 0.0% |

Measured with one node active at a time, so these are properties of the
`(node, cut)` pair alone.

**The question this table answers:** does the winning cut differ between tiers?
It does not. `k0` is the argmin on all three nodes, and not narrowly - 5.1x on
tier A, 15.5x on tier B, and tier C has no feasible split to compare against.
The winning cut is a property of the model and the transport here, not of the
device; section 6 takes that apart.

### 5.3 Where the time actually goes

![Stage breakdown](analysis/out/stage_breakdown.png)

<!-- T3 - generated by analysis/figures.py; regenerate with `make figures` -->

| node | cut | device | uplink | queue | server | residual |
|---|---|---|---|---|---|---|
| 11 (A) | k0 | 19.4 | 7.6 | 0.1 | 4.3 | 1.1 |
| 11 (A) | k_deep | 180.6 | 7.0 | 0.1 | 1.1 | 1.2 |
| 11 (A) | k_shallow | 154.5 | 9.7 | 0.1 | 1.5 | 1.1 |
| 21 (B) | k0 | 22.3 | 8.5 | 0.1 | 4.1 | 1.1 |
| 21 (B) | k_deep | 708.1 | 9.1 | 0.1 | 1.3 | 1.3 |
| 21 (B) | k_shallow | 543.9 | 11.0 | 0.1 | 1.6 | 1.2 |
| 31 (C) | k0 | 24.3 | 11.9 | 0.1 | 4.2 | 1.1 |

This is the diagnostic table: it says *why* a cut wins or loses, not just that
it does. What it shows:

- **`device` decides everything.** It is 154.5 ms for `k_shallow` on tier A and
  708.1 ms for `k_deep` on tier B, against a `server` term of 1.1-1.6 ms and an
  `uplink` of 7-11 ms. Nothing else is within two orders of magnitude.
- **`uplink` is nearly constant across cuts** - 7.0 to 11.9 ms - even though the
  payload varies 5x between `k_deep` (888 B) and `k_shallow` (4,668 B). At these
  sizes airtime is per-packet overhead, not bytes, so the data-inflation effect
  of section 2 costs almost nothing in time.
- **`k0` is the only cut where `server` is visible** at 4.1-4.3 ms, since its
  tail runs the whole model plus a JPEG decode. Even so it is a tenth of `k0`'s
  total, most of which is the SPIFFS read.
- **`queue` is 0.1 ms everywhere**, so per-node numbers in this table can be
  read independently: with one request in flight per node and a 3.45 ms tail,
  three nodes rarely collide. The coupling that does exist shows up in Table 5
  as end-to-end cost rather than as queueing here.

### 5.4 Policy comparison

![Latency CDFs per policy](analysis/out/policy_cdfs.png)

<!-- T4 - generated by analysis/figures.py; regenerate with `make figures` -->

| policy | node | cut actually run | n OK | mean ms | p95 ms |
|---|---|---|---|---|---|
| k0 (no split) | 11 (A) | k0 | 200 | 42.4 | 67.4 |
| k0 (no split) | 21 (B) | k0 | 200 | 37.2 | 61.5 |
| k0 (no split) | 31 (C) | k0 | 200 | 45.2 | 74.2 |
| k_shallow | 11 (A) | k_shallow | 200 | 174.1 | 207.2 |
| k_shallow | 21 (B) | k_shallow | 200 | 557.4 | 579.3 |
| k_shallow | 31 (C) | k0 | 200 | 42.6 | 71.0 |
| k_deep | 11 (A) | k_deep | 200 | 193.5 | 213.2 |
| k_deep | 21 (B) | k_deep | 200 | 718.3 | 741.3 |
| k_deep | 31 (C) | k0 | 200 | 44.5 | 71.1 |
| best static | 11 (A) | k0 | 200 | 45.8 | 62.5 |
| best static | 21 (B) | k0 | 200 | 36.6 | 64.0 |
| best static | 31 (C) | k0 | 200 | 42.5 | 72.0 |

Measured with the whole fleet active, so each column reflects that policy's
fleet composition. Compare policies to each other here; compare split points
in isolation in Table 2. The gap between a node's number here and its Table-2
number for the same cut is the cost of sharing the server tail.

The CDFs matter more than the means. Two policies with the same mean can have
very different tails, and for a fleet doing repeated inference the tail is what
determines the achievable rate. Look specifically at whether the winning policy
also wins at p95, or whether it trades a better mean for a worse tail.

### 5.5 The cost of running the fleet concurrently

<!-- T5 - generated by analysis/figures.py; regenerate with `make figures` -->

| policy | node | cut | isolated (T2) | concurrent (T4) | difference |
|---|---|---|---|---|---|
| k0 (no split) | 11 (A) | k0 | 32.6 | 42.4 | +30.2% |
| k0 (no split) | 21 (B) | k0 | 36.0 | 37.2 | +3.3% |
| k0 (no split) | 31 (C) | k0 | 41.5 | 45.2 | +8.8% |
| k_shallow | 11 (A) | k_shallow | 166.8 | 174.1 | +4.4% |
| k_shallow | 21 (B) | k_shallow | 557.8 | 557.4 | -0.1% |
| k_shallow | 31 (C) | k0 | 41.5 | 42.6 | +2.5% |
| k_deep | 11 (A) | k_deep | 190.0 | 193.5 | +1.8% |
| k_deep | 21 (B) | k_deep | 719.9 | 718.3 | -0.2% |
| k_deep | 31 (C) | k0 | 41.5 | 44.5 | +7.2% |
| best static | 11 (A) | k0 | 32.6 | 45.8 | +40.7% |
| best static | 21 (B) | k0 | 36.0 | 36.6 | +1.7% |
| best static | 31 (C) | k0 | 41.5 | 42.5 | +2.3% |

The same node running the same cut, measured alone in the sweep versus with the
whole fleet active. The difference is what sharing one server tail costs.

This is the column to watch for the argument the whole project rests on. A
policy whose server work is expensive - `k0` runs the entire model plus a JPEG
decode on the host - does not only pay for itself, it makes every other node
wait.
If that shows up here, then **the case for splitting is stronger at fleet scale
than the isolated numbers of Table 2 suggest**, because splitting moves work off
the one resource every node is queueing for.

**It is there, and with the asymmetry the argument predicts.** On node 11,
`k0` costs +30.2% and `best` (which is `k0`) +40.7% once the fleet runs
together, while `k_shallow` costs +4.4% and `k_deep` +1.8%. Policies that lean
on the shared tail pay an order of magnitude more for concurrency than policies
that do their work on the device.

A simulation of this fleet predicted the same shape before the hardware existed
(+13% to +44% for `k0` against -0.4% to +2.7% for the split cuts), which is
some evidence that the mechanism is understood rather than stumbled upon.

What it does not do is change the ranking. `k0` at 42.4 ms with its 30% penalty
still beats `k_shallow` at 174.1 ms by 4.1x. The fleet-scale case for splitting
is real, measurable, and about 25x too small here - see section 6.

### 5.6 Feasibility: which split points each tier can host

| Tier | Board | Arena source | `k_shallow` | `k_deep` | Feasible cuts |
|---|---|---|---|---|---|
| A | ESP32-S3-WROOM-1 N16R8 | 8 MB octal PSRAM @ 80 MHz | 149,972 B | 158,120 B | `k0`, `k_shallow`, `k_deep` |
| B | ESP32-CAM (ESP32-D0WD) | 8 MB PSRAM @ 40 MHz (4 MB mapped) | 142,296 B | 150,396 B | `k0`, `k_shallow`, `k_deep` |
| C | ESP32-D0WD-V3, no PSRAM | 100 KB internal static | **needs 138,240 B, has 98,480** | not linked | **`k0` only** |

Measured from each board's boot log (`ARENA cut=... used=...`), all three at
240 MHz. Two results worth stating plainly.

**Tier C cannot split at all.** `k_shallow` asks for 135 KB of arena and the
plain ESP32 offers 96 KB - short by 39 KB, not by a rounding error. This is the
outcome the design anticipated: on the weakest hardware the only rational
choice is not to split, and the fleet's heterogeneity lives in tiers A and B.
It also means tier C's row in every latency table is a single policy, and that
`k_deep`'s per-tier clamp (section 4.1) does real work.

**The same cut costs different arena on different silicon.** `k_shallow` needs
149,972 B on the S3 against 142,296 B on the ESP32 - 7.7 KB more for identical
weights and identical tensor shapes. The difference is the kernel library: the
S3 build uses esp-nn's vector paths, which allocate their own scratch. Worth
knowing before assuming an arena figure ports across a fleet.

### 5.7 What a slower server would change

![projected latency vs server speed](analysis/out/server_sensitivity.png)

<!-- T6 - generated by analysis/figures.py; regenerate with `make figures` -->

| node | best cut on this server | cut that takes over | server slower by | k0 tail cost at crossover (ms) |
|---|---|---|---|---|
| 11 (A) | k0 | k_shallow | 51x | 218 |
| 21 (B) | k0 | k_shallow | 217x | 881 |
| 31 (C) | k0 | - | no crossover | - |

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
| sweep | 1400 | 1400 | 0 | **0** | 0 | 0 |
| k0 | 603 | 600 | 0 | **0** | 3 | 0 |
| k_shallow | 610 | 600 | 9 | **0** | 1 | 0 |
| k_deep | 600 | 600 | 0 | **0** | 0 | 0 |
| best | 600 | 600 | 0 | **0** | 0 | 0 |

**Zero CRC failures across 3,813 requests**, and no node rebooted during any
run. Worst failure rate 1.6% (`k_shallow`), against the 3% the sample-count gate
allows; the sweep and two of the four policy runs were perfect.

Getting here took three protocol fixes that only hardware exposed. The device
blocked for exactly the task-watchdog timeout waiting for a `RESULT`, so any
request that never came back rebooted the node. A busy device silently dropped
an `ASSIGN` for a different request, so a single lost `RESULT` cost three
unacked assigns and a `LOST` row. And the device's idle timeout outlived the
orchestrator's own deadline, keeping it deaf after the server had moved on. All
three were invisible in simulation, because in simulation nothing is ever lost.
The 9 `TIMEOUT` rows in `k_shallow` are the residue: requests where the tensor
genuinely did not arrive within `t_max_ms`, which the protocol now handles by
abandoning rather than by rebooting.

## 6. Discussion

### Full offload wins on every node, and it is not close

| node | best cut | next best | margin |
|---|---|---|---|
| 11 (A) ESP32-S3 | `k0` 32.6 ms | `k_shallow` 166.8 ms | **5.1x** |
| 21 (B) ESP32-CAM | `k0` 36.0 ms | `k_shallow` 557.8 ms | **15.5x** |
| 31 (C) plain ESP32 | `k0` 41.5 ms | *no feasible split* | - |

No intermediate split beats sending the JPEG, on any tier, at either the mean
or p95. The sweep's argmin is `k0` for all three nodes, and the `best` policy -
which replays that choice - is indistinguishable from `k0`, as it must be.

**A single static policy is sufficient for this model on this fleet.** That is
a real answer to the question the project asked, and the 1,400-request sweep is
the evidence. It also means the orchestration machinery this repository builds
has, on this hardware, nothing to decide.

### Why: the split point never gets to matter

Table 3 makes the reason unambiguous. `device` dominates every split cut and
nothing else comes close:

| node | cut | device | uplink | queue | server |
|---|---|---|---|---|---|
| 11 (A) | `k_shallow` | **154.5** | 9.7 | 0.1 | 1.5 |
| 21 (B) | `k_deep` | **708.1** | 9.1 | 0.1 | 1.3 |

Splitting trades server work for device work and transfer for compute. On this
fleet the exchange rate is terrible: the head costs 121-668 ms on the MCU to
save 2.7-3.0 ms on the server, over a link where the whole transfer is 7-12 ms
regardless of which cut produced it. The uplink term barely moves between cuts
(7.0 to 11.0 ms across every row) even though the payload varies 5x, because at
these sizes airtime is dominated by per-packet overhead, not bytes.

That also explains the counter-intuitive ordering: **`k_deep` is worse than
`k_shallow` on both capable tiers** - 190.0 vs 166.8 ms on tier A, 719.9 vs
557.8 ms on tier B. Cutting deeper does shrink the tensor, from 4,668 B to 888 B
(below even the JPEG, exactly as the architecture predicts in Table 1), but it
buys 2.7 ms of transfer with 26-164 ms of extra inference. The data-inflation
effect this study was designed to expose is real and measurable, and it is
irrelevant, because transfer is not the binding constraint.

### The tiers differ enormously, and it changes nothing

Tier B needs 4.2x tier A's time for identical work - 504 ms against 121 ms for
`k_shallow` - despite the same clock, the same model, and the same quantization.
The ESP32-S3's vector extensions and its faster PSRAM are worth more than four
ESP32s. The feasibility sets differ too: tier C cannot host any head at all
(section 5.6), so on the weakest hardware the only available policy is the one that
wins anyway.

So the heterogeneity is real and large, and the correct policy is nevertheless
identical on all three tiers. The premise that motivates per-device split
selection - that different devices want different cuts - does not hold here.

### Fleet coupling is visible, and too small to matter

Table 5 was built to test whether splitting pays at fleet scale by taking work
off the shared server tail. The effect is there, in the predicted direction and
with the predicted asymmetry:

| policy on node 11 | isolated | concurrent | cost |
|---|---|---|---|
| `k0` | 32.6 | 42.4 | **+30.2%** |
| `best` (= `k0`) | 32.6 | 45.8 | **+40.7%** |
| `k_shallow` | 166.8 | 174.1 | +4.4% |
| `k_deep` | 190.0 | 193.5 | +1.8% |

Policies that lean on the server pay 30-41% when the fleet runs together;
policies that do their work on the device pay 2-4%. Splitting genuinely does
buy immunity to server contention. But `k0` at 42.4 ms with the penalty still
beats `k_shallow` at 174.1 ms by 4.1x. Three nodes and one server worker are
not enough contention to reverse a 5x compute deficit - and section 5.7 quantifies how
far from enough: the server would have to be 51x slower.

### What would have to change

`k_shallow` overtakes `k0` on tier A when the server's tail reaches ~218 ms
(section 5.7). For scale, a Raspberry Pi 4 - the server this study was originally
designed around - runs this model in roughly 10-25 ms. Splitting would still
have lost there, by about an order of magnitude in required server slowness.

So the conclusion does not depend on the substitute server, and it generalises
in a specific direction: **split inference pays when device compute is cheap
relative to the server, and MobileNetV2-0.35 on an ESP32 is the opposite
regime.** The head is 12 of 67 operations and costs 121 ms on the best board in
the fleet; the whole model costs 3.45 ms on a 400 MHz x86 core. Any argument for
splitting this workload has to start by finding a server two orders of magnitude
slower, or a device two orders of magnitude faster.

Where splitting would plausibly win, on this evidence: a much larger model whose
server-side cost is seconds rather than milliseconds, a link far slower than
20 MHz 802.11n, or a device with hardware NN acceleration that changes the
121 ms head cost. None of those is a small extrapolation from what was measured
here, and this report does not make them.

## 7. Threats to validity

- **The server is far faster than the edge server this study was designed
  around, and this decides part of the result.** The design called for a
  Raspberry Pi 4; that board is no longer available, so the x86 workstation that
  builds the firmware also runs the AP, the orchestrator and the tail. Capping
  the CPU to 400 MHz closes part of the gap - 3.45 / 1.36 / 0.76 ms per request
  for `k0` / `k_shallow` / `k_deep` against 0.29 / 0.13 / 0.07 ms uncapped
  (`make bench-tail`, four pinned cores) - but not all of it. Two consequences,
  both structural, now smaller than they were but not gone:
  1. **The `server` and `queue` stages stay small.** At 3.45 ms, `k0`'s tail is
     now a visible term next to a multi-millisecond `uplink`, and three nodes
     queueing behind one worker can produce measurable contention - Table 5 may
     have something in it after all, where at 0.29 ms it could not. But the
     server is still fast enough that a flat Table 5 would be a property of
     this server rather than a refutation of fleet coupling.
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

  Two mitigations were tried. Capping the CPU frequency to 400 MHz (the
  `intel_pstate` floor) is a real, smooth slowdown, is used for the session,
  and is what the numbers above were measured under. A
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
- **The SPIFFS read is not free, and is not quite equal across policies.**
  `k0` reads a ~1.8 KB JPEG, every split cut reads the 27,648 B raw tensor.
  Measured, that is 17-37 ms against 32-41 ms: 15x the bytes for ~1.4x the
  time, because SPIFFS cost here is dominated by `fopen`, not throughput. So
  the split cuts carry roughly 10 ms of extra read that a camera deployment
  would not pay, against head inference of 121-546 ms. It is reported in the
  `device` column rather than removed, and it is far too small to change any
  ordering.
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
