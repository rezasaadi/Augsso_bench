# AugSSO (Rust) + Benchmark Harness

This repo contains a **self-contained AugSSO implementation** (client + threshold identity servers) and a
single benchmark binary that reports the same categories as the uploaded benchmark suites:

- **Client protocol time** (`--kind proto`) – setup, registration, authentication, verify
- **Server protocol time** (`--kind sp`) – store, db_get, respond1, respond2 (+ a combined respond_total estimate)
- **Crypto primitive microbenches** (`--kind prim`) – pairings, Shamir/Lagrange combine, PKE, AEAD
- **Network-only simulation** (`--kind net`) – LAN/WAN model (latency + jitter + bandwidth)
- **End-to-end simulation** (`--kind full`) – client CPU + simulated net + injected server p50

## Layout

```
src/
  crypto_core.rs           # Blake3 + XChaCha20-Poly1305 (SE)
  crypto_augsso.rs         # BLS12-381 glue + Shamir helpers + benchmark-friendly PKE (X25519+XChaCha)
  protocols/augsso.rs      # AugSSO protocol implementation + fixtures
  bin/bench_augsso.rs      # benchmark driver
```

## Build

```bash
cargo build --release
```

## Run benchmarks

Default runs everything (proto + prim + sp + net + full) for `nsp=20,40,60` and `tsp=ceil(50% of nsp)`:

```bash
cargo run --release --bin bench_augsso
```

Client protocol only:

```bash
cargo run --release --bin bench_augsso -- --kind proto
```

Primitives only:

```bash
cargo run --release --bin bench_augsso -- --kind prim
```

Network simulation only (LAN+WAN):

```bash
cargo run --release --bin bench_augsso -- --kind net --net all
```

Full end-to-end on WAN, include RNG cost in timed sections:

```bash
cargo run --release --bin bench_augsso -- --kind full --net wan --rng-in-timed
```

Change the server counts / thresholds:

```bash
cargo run --release --bin bench_augsso -- --nsp 50,100 --tsp 10,20
# or derive tsp from percentages
cargo run --release --bin bench_augsso -- --nsp 50,100 --tsp-pct 20,40,60
```

Output file:

```bash
cargo run --release --bin bench_augsso -- --out augsso_bench.txt
```

## Notes on protocol modeling choices

- **Threshold token**: implemented as threshold BLS-style signatures on `H(pld)` over BLS12-381.
- **Authentication request payload (`pld`)**: modeled as `pld = x || idU` (64 bytes), matching the “payload + identity binding” pattern used in the other uploaded benches.
- **PKE**: benchmark-friendly CPA-secure construction for `PKE.Enc/Dec` using **X25519** (key agreement) + **XChaCha20-Poly1305** (AEAD).
- **Hash-to-G1**: a deterministic map `Blake3-XOF -> Scalar -> scalar_mul(G1_gen)` (same style as the other benches; not IETF hash-to-curve).

## Output columns

Each line is:

```
scheme kind op rng_in_timed nsp tsp samples warmup min_ns p50_ns p95_ns max_ns mean_ns stddev_ns
```

Times are in **nanoseconds**.
