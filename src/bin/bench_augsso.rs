#![allow(clippy::needless_range_loop)]
use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::time::Instant;
use std::time::Duration;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use augsso_bench::protocols::augsso;
use augsso_bench::{crypto_augsso as ac, crypto_core};

// Stats helpers 

#[derive(Clone, Debug)]
struct Stats {
    n: usize,
    min_ns: u128,
    p50_ns: u128,
    p95_ns: u128,
    max_ns: u128,
    mean_ns: f64,
    stddev_ns: f64,
}

fn compute_stats(mut xs: Vec<u128>) -> Stats {
    xs.sort_unstable();
    let n = xs.len();
    let min_ns = xs[0];
    let p50_ns = xs[n / 2];
    let p95_ns = xs[(n * 95) / 100];
    let max_ns = xs[n - 1];

    let sum: u128 = xs.iter().sum();
    let mean_ns = (sum as f64) / (n as f64);

    let mut var = 0.0;
    for &x in &xs {
        let d = (x as f64) - mean_ns;
        var += d * d;
    }
    let stddev_ns = if n > 1 { (var / ((n - 1) as f64)).sqrt() } else { 0.0 };

    Stats {
        n,
        min_ns,
        p50_ns,
        p95_ns,
        max_ns,
        mean_ns,
        stddev_ns,
    }
}

fn bench_u128(mut f: impl FnMut() -> u128, warmup: usize, samples: usize) -> Stats {
    for _ in 0..warmup {
        black_box(f());
    }
    let mut xs = Vec::with_capacity(samples);
    for _ in 0..samples {
        xs.push(f());
    }
    compute_stats(xs)
}

fn time_call_ns<R>(mut f: impl FnMut() -> R) -> u64 {
    let t0 = Instant::now();
    let out = f();
    black_box(out);
    t0.elapsed().as_nanos() as u64
}

fn median_ns(mut xs: Vec<u64>) -> u64 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn write_header(out: &mut BufWriter<File>) -> std::io::Result<()> {
    writeln!(
        out,
        "scheme kind op rng_in_timed nsp tsp samples warmup min_ns p50_ns p95_ns max_ns mean_ns stddev_ns"
    )
}

fn write_row(
    out: &mut BufWriter<File>,
    scheme: &str,
    kind: &str,
    op: &str,
    rng_in_timed: bool,
    nsp: usize,
    tsp: usize,
    warmup: usize,
    st: &Stats,
) -> std::io::Result<()> {
    writeln!(
        out,
        "{} {} {} {} {} {} {} {} {} {} {} {} {:.3} {:.3}",
        scheme,
        kind,
        op,
        if rng_in_timed { 1 } else { 0 },
        nsp,
        tsp,
        st.n,
        warmup,
        st.min_ns,
        st.p50_ns,
        st.p95_ns,
        st.max_ns,
        st.mean_ns,
        st.stddev_ns,
    )
}

// Network model 

#[derive(Clone, Copy, Debug)]
struct NetProfile {
    name: &'static str,
    one_way_ns: u64,
    jitter_ns: u64,
    bw_bps: u64,
    overhead_bytes: usize,
}

fn mk_profile(name: &'static str, rtt_ms: f64, jitter_ms: f64, bw_mbps: f64, overhead_bytes: usize) -> NetProfile {
    let one_way_ns = ((rtt_ms * 1_000_000.0) / 2.0).round() as u64;
    let jitter_ns = (jitter_ms * 1_000_000.0).round() as u64;
    let bw_bps = (bw_mbps * 1_000_000.0).round() as u64;
    NetProfile {
        name,
        one_way_ns,
        jitter_ns,
        bw_bps,
        overhead_bytes,
    }
}

fn tx_time_ns(bytes: usize, bw_bps: u64) -> u64 {
    let bits = (bytes as u64) * 8;
    let num = bits.saturating_mul(1_000_000_000u64);
    (num + bw_bps - 1) / bw_bps
}

fn sample_jitter_ns(jitter_ns: u64, rng: &mut impl RngCore) -> i64 {
    if jitter_ns == 0 {
        return 0;
    }
    let r = (rng.next_u64() as f64) / (u64::MAX as f64);
    let x = (2.0 * r - 1.0) * (jitter_ns as f64);
    x.round() as i64
}

fn add_signed_ns(t: u64, delta: i64) -> u64 {
    if delta >= 0 {
        t.saturating_add(delta as u64)
    } else {
        t.saturating_sub((-delta) as u64)
    }
}


/// This matches the model used by the PAS-TA-U / PASTA benches:
/// - client uplink serialization
/// - one-way latency + jitter
/// - server p50 processing time
/// - downlink serialization
fn simulate_parallel_phase(
    k: usize,
    req_bytes_per_server: usize,
    resp_bytes_per_server: usize,
    server_proc_ns_p50: u64,
    prof: NetProfile,
    rng: &mut impl RngCore,
) -> u64 {
    if k == 0 {
        return 0;
    }

    let req_bytes = req_bytes_per_server + prof.overhead_bytes;
    let resp_bytes = resp_bytes_per_server + prof.overhead_bytes;

    let tx_req = tx_time_ns(req_bytes, prof.bw_bps);
    let tx_resp = tx_time_ns(resp_bytes, prof.bw_bps);

    let mut server_start_times = Vec::with_capacity(k);
    for i in 0..k {
        let send_done = (i as u64 + 1) * tx_req;
        let jitter = sample_jitter_ns(prof.jitter_ns, rng);
        let arrive = add_signed_ns(send_done.saturating_add(prof.one_way_ns), jitter);
        server_start_times.push(arrive);
    }

    let mut server_done_times = Vec::with_capacity(k);
    for &st in &server_start_times {
        server_done_times.push(st.saturating_add(server_proc_ns_p50));
    }

    // Downlink
    server_done_times.sort_unstable();

    let mut client_rx_done = 0u64;
    for (i, &done) in server_done_times.iter().enumerate() {
        let jitter = sample_jitter_ns(prof.jitter_ns, rng);
        let first_bit = add_signed_ns(done.saturating_add(prof.one_way_ns), jitter);

        // serialization starts when both (a) response has started arriving and (b) link is free
        let start = client_rx_done.max(first_bit);
        let end = start.saturating_add(tx_resp);
        client_rx_done = end;

        black_box(i);
    }

    client_rx_done
}

// Scheme-specific sizes 

const REQ1_BYTES_PER_SERVER: usize = augsso::PLD_LEN + augsso::BPW_LEN; 
const RESP1_BYTES_PER_SERVER: usize = augsso::SIGMA_LEN + augsso::CTRES_LEN; 
const REQ2_BYTES_PER_SERVER: usize = augsso::CTCH_LEN; 
const RESP2_BYTES_PER_SERVER: usize = augsso::CTTKN_LEN; 

// Benchmark helpers

fn seed_for(tag: &[u8], nsp: usize, tsp: usize) -> [u8; 32] {
    use blake3;
    let mut h = blake3::Hasher::new();
    h.update(tag);
    h.update(&(nsp as u64).to_le_bytes());
    h.update(&(tsp as u64).to_le_bytes());
    let out = h.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(out.as_bytes());
    seed
}

#[derive(Clone, Copy, Debug)]
struct ServerProcP50 {
    respond1_ns: u64,
    respond2_ns: u64,
    respond_total_ns: u64,
    db_get_ns: u64,
    store_ns: u64,
}

fn measure_server_procs_p50(
    nsp: usize,
    tsp: usize,
    warmup: usize,
    samples: usize,
    rng_in_timed: bool,
) -> ServerProcP50 {
    let fx = augsso::make_fixture(nsp, tsp);
    let srv = &fx.servers[0];

    let mut rng_it = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proc/it", nsp, tsp));
    let it = augsso::make_iter_data(&fx, &mut rng_it);
    let (_st_client, creq) = augsso::request_with_r(&fx.password, fx.pld, &fx.t_set, it.r);

    // --- respond1 ---
    let mut rng_resp1 = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proc/respond1_rng", nsp, tsp));
    for _ in 0..warmup {
        let out = if rng_in_timed {
            augsso::respond1(srv, &creq, &mut rng_resp1)
        } else {
            augsso::respond1_with_rand(srv, &creq, &it.r1[0])
        };
        black_box(out);
    }
    let mut xs = Vec::with_capacity(samples);
    for s in 0..samples {
        xs.push(time_call_ns(|| {
            let out = if rng_in_timed {
                augsso::respond1(srv, &creq, &mut rng_resp1)
            } else {
                augsso::respond1_with_rand(srv, &creq, &it.r1[0])
            };
            black_box((s, out))
        }));
    }
    let respond1_ns = median_ns(xs);

    // Prepare session + ctch for respond2 benchmarks.
    let (_resp1, sess) = augsso::respond1_with_rand(srv, &creq, &it.r1[0]).unwrap();
    let ctch = {
        let aad: [u8; 0] = [];
        augsso_bench::crypto_augsso::xchacha_encrypt_detached_with_nonce::<{augsso::CTCH_PT_LEN}>(
            &sess.ek,
            &aad,
            &sess.ch,
            &it.ctch_nonces[0],
        )
    };

    // --- respond2 ---
    let mut rng_resp2 = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proc/respond2_rng", nsp, tsp));
    for _ in 0..warmup {
        let out = if rng_in_timed {
            augsso::respond2(srv, &sess, &ctch, &mut rng_resp2)
        } else {
            augsso::respond2_with_nonce(srv, &sess, &ctch, &it.cttkn_nonces[0])
        };
        black_box(out);
    }
    let mut xs = Vec::with_capacity(samples);
    for s in 0..samples {
        xs.push(time_call_ns(|| {
            let out = if rng_in_timed {
                augsso::respond2(srv, &sess, &ctch, &mut rng_resp2)
            } else {
                augsso::respond2_with_nonce(srv, &sess, &ctch, &it.cttkn_nonces[0])
            };
            black_box((s, out))
        }));
    }
    let respond2_ns = median_ns(xs);

    // Total server work for one full auth session (respond1 + respond2)
    let respond_total_ns = respond1_ns.saturating_add(respond2_ns);

    // --- DB get ---
    for _ in 0..warmup {
        black_box(srv.has_record(fx.c));
    }
    let mut xs = Vec::with_capacity(samples);
    for _ in 0..samples {
        xs.push(time_call_ns(|| {
            let v = srv.has_record(fx.c);
            black_box(v)
        }));
    }
    let db_get_ns = median_ns(xs);

    // --- Store ---
    let msg = {
        let mut rng = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proc/store_msg", nsp, tsp));
        augsso::registration(nsp, tsp, &fx.password, &mut rng).msgs[0].clone()
    };

    let mut srv2 = augsso::AugSsoServer::new(
        1,
        fx.servers[0].gamma_i,
        fx.servers[0].phi_i,
        fx.servers[0].gamma_pub_i,
        fx.servers[0].phi_pub_i,
        fx.servers[0].pp,
    );

    for _ in 0..warmup {
        srv2.store(fx.c, &msg);
    }
    let mut xs = Vec::with_capacity(samples);
    for _ in 0..samples {
        xs.push(time_call_ns(|| {
            srv2.store(fx.c, &msg);
        }));
    }
    let store_ns = median_ns(xs);

    ServerProcP50 {
        respond1_ns,
        respond2_ns,
        respond_total_ns,
        db_get_ns,
        store_ns,
    }
}

//client protocol phases


fn bench_client_proto(
    nsp: usize,
    tsp: usize,
    warmup: usize,
    samples: usize,
    rng_in_timed: bool,
    out: &mut BufWriter<File>,
) -> std::io::Result<()> {
    let scheme = "augsso";
    let fx = augsso::make_fixture(nsp, tsp);

    // setup
    {
        let mut rng = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proto/setup_rng", nsp, tsp));
        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let outv = augsso::setup(nsp, tsp, &mut rng);
                black_box(outv);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "proto", "setup", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // reg
    {
        let mut rng = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proto/reg_rng", nsp, tsp));
        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let outv = augsso::registration(nsp, tsp, &fx.password, &mut rng);
                black_box(outv);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "proto", "reg", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // auth: request + client phase1 + finalize, excluding server processing
    {
        let mut rng_it =
            ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proto/auth_it", nsp, tsp));
        let mut rng_req =
            ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proto/auth_req_rng", nsp, tsp));
        let mut rng_cli =
            ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proto/auth_cli_rng", nsp, tsp));
        let mut rng_srv =
            ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proto/auth_srv_rng", nsp, tsp));

        // helper that executes one iteration of the auth benchmark,
        // returning `Some(ns)` on success or `None` if we failed all
        // retry attempts during client phase1.
        let mut one_iter = || -> Option<u128> {
            let it = augsso::make_iter_data(&fx, &mut rng_it);

            // request (timed)
            let t0 = Instant::now();
            let (st_client, creq) = if rng_in_timed {
                augsso::request(&fx.password, fx.pld, &fx.t_set, &mut rng_req)
            } else {
                augsso::request_with_r(&fx.password, fx.pld, &fx.t_set, it.r)
            };
            let t_req = t0.elapsed();

            // server responses phase1 (not timed)
            let mut resps1 = Vec::with_capacity(tsp);
            let mut sess = Vec::with_capacity(tsp);
            for j in 0..tsp {
                let srv = &fx.servers[j];
                let (r1, s1) = if rng_in_timed {
                    augsso::respond1(srv, &creq, &mut rng_srv).unwrap()
                } else {
                    augsso::respond1_with_rand(srv, &creq, &it.r1[j]).unwrap()
                };
                resps1.push(r1);
                sess.push(s1);
            }

            // client phase1 with retry loop
            let (t_phase1, phase1) = {
                let mut t_phase1 = Duration::from_nanos(0);
                let mut out = None;
                for _ in 0..8 {
                    let t1 = Instant::now();
                    let attempt = augsso::client_phase1(
                        &st_client,
                        &creq,
                        &resps1,
                        &fx.pk_shares,
                        if rng_in_timed { None } else { Some(&it.ctch_nonces) },
                        &mut rng_cli,
                    );
                    t_phase1 = t1.elapsed();
                    if let Some(ph) = attempt {
                        out = Some((t_phase1, ph));
                        break;
                    }
                }
                match out {
                    Some(v) => v,
                    None => return None,
                }
            };

            // server responses phase2 (not timed)
            let mut resps2 = Vec::with_capacity(tsp);
            for j in 0..tsp {
                let sid = (j + 1) as u32;
                let ctch = phase1.ct_ch.iter().find(|(id, _)| *id == sid).unwrap().1.clone();
                let srv = &fx.servers[j];
                let outv = if rng_in_timed {
                    augsso::respond2(srv, &sess[j], &ctch, &mut rng_srv).unwrap()
                } else {
                    augsso::respond2_with_nonce(
                        srv,
                        &sess[j],
                        &ctch,
                        &it.cttkn_nonces[j],
                    )
                    .unwrap()
                };
                resps2.push(outv);
            }

            // finalize (timed)
            let t2 = Instant::now();
            let tk = augsso::client_finalize(
                &st_client,
                &phase1,
                &resps2,
                &fx.gamma_pub_shares,
            )
            .unwrap();
            black_box(tk);
            let t_fin = t2.elapsed();

            Some((t_req + t_phase1 + t_fin).as_nanos())
        };

        let mut samples_vec = Vec::with_capacity(samples);

        // warmup: run and discard (ignore failures)
        for _ in 0..warmup {
            let _ = one_iter();
        }

        // measurement: retry on phase‑1 failures
        // (but don't hang forever if something is fundamentally broken).
        for _ in 0..samples {
            let mut tries = 0usize;
            loop {
                if let Some(ns) = one_iter() {
                    samples_vec.push(ns);
                    break;
                }
                tries += 1;
                if tries >= 256 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!(
                            "auth iteration failed repeatedly (nsp={}, tsp={}). this usually means client couldn't decrypt CTres (hpw mismatch) or pairing checks failed.",
                            nsp, tsp
                        ),
                    ));
                }
            }
        }

        let st = compute_stats(samples_vec);
        write_row(
            out,
            scheme,
            "proto",
            "auth",
            rng_in_timed,
            nsp,
            tsp,
            warmup,
            &st,
        )?;
    }

    // verify 
    {
        // Build one token deterministically (not timed)
        let mut rng_it = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proto/verify_it", nsp, tsp));
        let mut rng_cli = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/proto/verify_cli_rng", nsp, tsp));
        let it = augsso::make_iter_data(&fx, &mut rng_it);
        let (st_client, creq) = augsso::request_with_r(&fx.password, fx.pld, &fx.t_set, it.r);

        let mut resps1 = Vec::with_capacity(tsp);
        let mut sess = Vec::with_capacity(tsp);
        for j in 0..tsp {
            let srv = &fx.servers[j];
            let (r1, s1) = augsso::respond1_with_rand(srv, &creq, &it.r1[j]).unwrap();
            resps1.push(r1);
            sess.push(s1);
        }
        let phase1 = augsso::client_phase1(
            &st_client,
            &creq,
            &resps1,
            &fx.pk_shares,
            Some(&it.ctch_nonces),
            &mut rng_cli,
        )
        .unwrap();
        let mut resps2 = Vec::with_capacity(tsp);
        for j in 0..tsp {
            let sid = (j + 1) as u32;
            let ctch = phase1.ct_ch.iter().find(|(id, _)| *id == sid).unwrap().1.clone();
            let srv = &fx.servers[j];
            let outv = augsso::respond2_with_nonce(srv, &sess[j], &ctch, &it.cttkn_nonces[j]).unwrap();
            resps2.push(outv);
        }
        let tk = augsso::client_finalize(&st_client, &phase1, &resps2, &fx.gamma_pub_shares).unwrap();

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let ok = augsso::verify_token(&fx.gamma_pk, &tk);
                black_box(ok);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );

        write_row(out, scheme, "proto", "verify", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    Ok(())
}

//server phases

fn bench_server_phases(
    nsp: usize,
    tsp: usize,
    warmup: usize,
    samples: usize,
    rng_in_timed: bool,
    out: &mut BufWriter<File>,
) -> std::io::Result<()> {
    let scheme = "augsso";
    let fx = augsso::make_fixture(nsp, tsp);
    let srv = &fx.servers[0];

    // db_get
    {
        let st = bench_u128(
            || {
                let ns = time_call_ns(|| srv.has_record(fx.c));
                ns as u128
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "sp", "db_get", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // store
    {
        let msg = {
            let mut rng = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/sp/store_msg", nsp, tsp));
            augsso::registration(nsp, tsp, &fx.password, &mut rng).msgs[0].clone()
        };

        let mut srv2 = augsso::AugSsoServer::new(
            1,
            fx.servers[0].gamma_i,
            fx.servers[0].phi_i,
            fx.servers[0].gamma_pub_i,
            fx.servers[0].phi_pub_i,
            fx.servers[0].pp,
        );

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                srv2.store(fx.c, &msg);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );

        write_row(out, scheme, "sp", "store", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // respond1
    {
        let mut rng_it = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/sp/respond1_it", nsp, tsp));
        let it = augsso::make_iter_data(&fx, &mut rng_it);
        let (_st_client, creq) = augsso::request_with_r(&fx.password, fx.pld, &fx.t_set, it.r);
        let mut rng_resp1 = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/sp/respond1_rng", nsp, tsp));

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let outv = if rng_in_timed {
                    augsso::respond1(srv, &creq, &mut rng_resp1)
                } else {
                    augsso::respond1_with_rand(srv, &creq, &it.r1[0])
                };
                black_box(outv);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );

        write_row(out, scheme, "sp", "respond1", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // respond2
    {
        let mut rng_it = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/sp/respond2_it", nsp, tsp));
        let it = augsso::make_iter_data(&fx, &mut rng_it);
        let (_st_client, creq) = augsso::request_with_r(&fx.password, fx.pld, &fx.t_set, it.r);
        let (_resp1, sess) = augsso::respond1_with_rand(srv, &creq, &it.r1[0]).unwrap();

        let ctch = {
            let aad: [u8; 0] = [];
            ac::xchacha_encrypt_detached_with_nonce::<{augsso::CTCH_PT_LEN}>(
                &sess.ek,
                &aad,
                &sess.ch,
                &it.ctch_nonces[0],
            )
        };

        let mut rng_resp2 = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/sp/respond2_rng", nsp, tsp));

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let outv = if rng_in_timed {
                    augsso::respond2(srv, &sess, &ctch, &mut rng_resp2)
                } else {
                    augsso::respond2_with_nonce(srv, &sess, &ctch, &it.cttkn_nonces[0])
                };
                black_box(outv);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );

        write_row(out, scheme, "sp", "respond2", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // respond_total (respond1 + respond2)
    {
        let proc = measure_server_procs_p50(nsp, tsp, warmup, samples, rng_in_timed);
        let xs = vec![proc.respond_total_ns as u128; samples.max(1)];
        let st = compute_stats(xs);
        write_row(out, scheme, "sp", "respond_total_p50sum", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    Ok(())
}

// primitives


fn bench_primitives(
    nsp: usize,
    tsp: usize,
    warmup: usize,
    samples: usize,
    rng_in_timed: bool,
    out: &mut BufWriter<File>,
) -> std::io::Result<()> {
    let scheme = "augsso";
    let fx = augsso::make_fixture(nsp, tsp);

    // H_pw -> G1
    {
        let pw = fx.password.clone();
        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let p = ac::hash_to_g1(b"augsso/H_pw/v1", &pw);
                black_box(p);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "hash_to_g1_pw", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // H_pld -> G1
    {
        let pld = fx.pld;
        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let p = ac::hash_to_g1(b"augsso/H_pld/v1", &pld);
                black_box(p);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "hash_to_g1_pld", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // Lagrange coeffs at zero
    {
        let ids: Vec<u32> = (1..=tsp as u32).collect();
        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let ls = ac::lagrange_coeffs_at_zero(&ids);
                black_box(ls);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "lagrange_coeffs", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // Combine G1 (t points)
    {
        let ids: Vec<u32> = (1..=tsp as u32).collect();
        let points: Vec<ac::G1> = (0..tsp)
            .map(|i| ac::hash_to_g1(b"augsso/prim/pt", &[(i as u8)]))
            .collect();
        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let p = ac::combine_g1_at_zero(&ids, &points);
                black_box(p);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "combine_g1", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // Pairing verify (BLS-style)
    {
        let pld = fx.pld;
        let h = ac::hash_to_g1(b"augsso/H_pld/v1", &pld);
        let pk = fx.gamma_pk;
        let sig = h * fx.servers[0].gamma_i; // one share (not valid for pk), but still triggers pairing
        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let ok = ac::pairing_check_sig(&h, &sig, &pk);
                black_box(ok);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "pairing_check", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // PKE.KG
    {
        let hpw = [42u8; 32];
        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let kp = ac::pke_kg(&hpw);
                black_box(kp);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "pke_kg", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // PKE.Enc (64B)
    {
        let hpw = [7u8; 32];
        let kp = ac::pke_kg(&hpw);
        let pt = [9u8; augsso::PKE_PT_LEN];
        let aad: [u8; 0] = [];
        let mut rng = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/prim/pke_enc_rng", nsp, tsp));
        let eph = [1u8; 32];
        let nonce = [2u8; crypto_core::NONCE_LEN];

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let ct = if rng_in_timed {
                    ac::pke_enc::<{augsso::PKE_PT_LEN}>(&kp.pk, &aad, &pt, &mut rng)
                } else {
                    ac::pke_enc_with_eph_and_nonce::<{augsso::PKE_PT_LEN}>(&kp.pk, &aad, &pt, &eph, &nonce)
                };
                black_box(ct);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "pke_encrypt", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // PKE.Dec (64B)
    {
        let hpw = [7u8; 32];
        let kp = ac::pke_kg(&hpw);
        let pt = [9u8; augsso::PKE_PT_LEN];
        let aad: [u8; 0] = [];
        let eph = [1u8; 32];
        let nonce = [2u8; crypto_core::NONCE_LEN];
        let ct = ac::pke_enc_with_eph_and_nonce::<{augsso::PKE_PT_LEN}>(&kp.pk, &aad, &pt, &eph, &nonce);

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let outv = ac::pke_dec::<{augsso::PKE_PT_LEN}>(&kp.sk, &aad, &ct);
                black_box(outv);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "pke_decrypt", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // AEAD enc (CTch: 32B)
    {
        let key = [3u8; 32];
        let aad: [u8; 0] = [];
        let pt = [4u8; augsso::CTCH_PT_LEN];
        let mut rng = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/prim/aead32_rng", nsp, tsp));
        let nonce = [5u8; crypto_core::NONCE_LEN];

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let ct = if rng_in_timed {
                    crypto_core::xchacha_encrypt_detached::<{augsso::CTCH_PT_LEN}>(&key, &aad, &pt, &mut rng)
                } else {
                    crypto_core::xchacha_encrypt_detached_with_nonce::<{augsso::CTCH_PT_LEN}>(&key, &aad, &pt, &nonce)
                };
                black_box(ct);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "aead_encrypt_32", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // AEAD dec (CTch)
    {
        let key = [3u8; 32];
        let aad: [u8; 0] = [];
        let pt = [4u8; augsso::CTCH_PT_LEN];
        let nonce = [5u8; crypto_core::NONCE_LEN];
        let ct = crypto_core::xchacha_encrypt_detached_with_nonce::<{augsso::CTCH_PT_LEN}>(&key, &aad, &pt, &nonce);

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let outv = crypto_core::xchacha_decrypt_detached::<{augsso::CTCH_PT_LEN}>(&key, &aad, &ct);
                black_box(outv);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "aead_decrypt_32", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // AEAD enc (CTtkn: 112B)
    {
        let key = [6u8; 32];
        let aad: [u8; 0] = [];
        let pt = [7u8; 112];
        let mut rng = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/prim/aead112_rng", nsp, tsp));
        let nonce = [8u8; crypto_core::NONCE_LEN];

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let ct = if rng_in_timed {
                    crypto_core::xchacha_encrypt_detached::<112>(&key, &aad, &pt, &mut rng)
                } else {
                    crypto_core::xchacha_encrypt_detached_with_nonce::<112>(&key, &aad, &pt, &nonce)
                };
                black_box(ct);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "aead_encrypt_112", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    // AEAD dec (CTtkn)
    {
        let key = [6u8; 32];
        let aad: [u8; 0] = [];
        let pt = [7u8; 112];
        let nonce = [8u8; crypto_core::NONCE_LEN];
        let ct = crypto_core::xchacha_encrypt_detached_with_nonce::<112>(&key, &aad, &pt, &nonce);

        let st = bench_u128(
            || {
                let t0 = Instant::now();
                let outv = crypto_core::xchacha_decrypt_detached::<112>(&key, &aad, &ct);
                black_box(outv);
                t0.elapsed().as_nanos()
            },
            warmup,
            samples,
        );
        write_row(out, scheme, "prim", "aead_decrypt_112", rng_in_timed, nsp, tsp, warmup, &st)?;
    }

    Ok(())
}

// ---------------------------
// Bench: network-only simulation
// ---------------------------

fn bench_net(
    nsp: usize,
    tsp: usize,
    warmup: usize,
    samples: usize,
    prof: NetProfile,
    out: &mut BufWriter<File>,
) -> std::io::Result<()> {
    let scheme = "augsso";
    let mut rng = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/net/rng", nsp, tsp));

    let st = bench_u128(
        || {
            let ns1 = simulate_parallel_phase(tsp, REQ1_BYTES_PER_SERVER, RESP1_BYTES_PER_SERVER, 0, prof, &mut rng);
            let ns2 = simulate_parallel_phase(tsp, REQ2_BYTES_PER_SERVER, RESP2_BYTES_PER_SERVER, 0, prof, &mut rng);
            (ns1 as u128) + (ns2 as u128)
        },
        warmup,
        samples,
    );

    write_row(
        out,
        scheme,
        "net",
        &format!("auth_{}", prof.name),
        false,
        nsp,
        tsp,
        warmup,
        &st,
    )
}

//full end-to-end (client CPU + simulated net + injected server p50)


fn bench_full(
    nsp: usize,
    tsp: usize,
    warmup: usize,
    samples: usize,
    rng_in_timed: bool,
    prof: NetProfile,
    proc_warmup: usize,
    proc_samples: usize,
    out: &mut BufWriter<File>,
) -> std::io::Result<()> {
    let scheme = "augsso";

    // Calibrate server p50 processing time (respond1/respond2)
    let proc = measure_server_procs_p50(nsp, tsp, proc_warmup, proc_samples, rng_in_timed);

    let fx = augsso::make_fixture(nsp, tsp);
    let mut rng_it = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/full/it", nsp, tsp));
    let mut rng_net = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/full/net", nsp, tsp));
    let mut rng_req = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/full/req_rng", nsp, tsp));
    let mut rng_cli = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/full/cli_rng", nsp, tsp));
    let mut rng_srv = ChaCha20Rng::from_seed(seed_for(b"bench_augsso/full/srv_rng", nsp, tsp));

    let st = bench_u128(
        || {
            let it = augsso::make_iter_data(&fx, &mut rng_it);

            // client: request (timed)
            let t0 = Instant::now();
            let (st_client, creq) = if rng_in_timed {
                augsso::request(&fx.password, fx.pld, &fx.t_set, &mut rng_req)
            } else {
                augsso::request_with_r(&fx.password, fx.pld, &fx.t_set, it.r)
            };
            let t_req = t0.elapsed();

            // server phase1 responses (not timed)
            let mut resps1 = Vec::with_capacity(tsp);
            let mut sess = Vec::with_capacity(tsp);
            for j in 0..tsp {
                let srv = &fx.servers[j];
                let (r1, s1) = if rng_in_timed {
                    augsso::respond1(srv, &creq, &mut rng_srv).unwrap()
                } else {
                    augsso::respond1_with_rand(srv, &creq, &it.r1[j]).unwrap()
                };
                resps1.push(r1);
                sess.push(s1);
            }

            // client phase1 (timed)
            let t1 = Instant::now();
            let phase1 = augsso::client_phase1(
                &st_client,
                &creq,
                &resps1,
                &fx.pk_shares,
                if rng_in_timed { None } else { Some(&it.ctch_nonces) },
                &mut rng_cli,
            )
            .unwrap();
            let t_phase1 = t1.elapsed();

            // server phase2 responses (not timed)
            let mut resps2 = Vec::with_capacity(tsp);
            for j in 0..tsp {
                let sid = (j + 1) as u32;
                let ctch = phase1.ct_ch.iter().find(|(id, _)| *id == sid).unwrap().1.clone();
                let srv = &fx.servers[j];
                let outv = if rng_in_timed {
                    augsso::respond2(srv, &sess[j], &ctch, &mut rng_srv).unwrap()
                } else {
                    augsso::respond2_with_nonce(
                        srv,
                        &sess[j],
                        &ctch,
                        &it.cttkn_nonces[j],
                    )
                    .unwrap()
                };
                resps2.push(outv);
            }

            // client finalize (timed)
            let t2 = Instant::now();
            let tk = augsso::client_finalize(
                &st_client,
                &phase1,
                &resps2,
                &fx.gamma_pub_shares,
            )
            .unwrap();
            black_box(tk);
            let t_fin = t2.elapsed();

            let client_ns = (t_req + t_phase1 + t_fin).as_nanos() as u64;

            // net + server proc
            let net1_ns = simulate_parallel_phase(
                tsp,
                REQ1_BYTES_PER_SERVER,
                RESP1_BYTES_PER_SERVER,
                proc.respond1_ns,
                prof,
                &mut rng_net,
            );
            let net2_ns = simulate_parallel_phase(
                tsp,
                REQ2_BYTES_PER_SERVER,
                RESP2_BYTES_PER_SERVER,
                proc.respond2_ns,
                prof,
                &mut rng_net,
            );

            (client_ns as u128) + (net1_ns as u128) + (net2_ns as u128)
        },
        warmup,
        samples,
    );

    write_row(
        out,
        scheme,
        "full",
        &format!("auth_{}", prof.name),
        rng_in_timed,
        nsp,
        tsp,
        warmup,
        &st,
    )
}

// ---------------------------
// CLI + main
// ---------------------------

fn parse_usize_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| x.trim().parse::<usize>().expect("bad integer"))
        .collect()
}

fn ceil_pct(n: usize, pct: usize) -> usize {
    // ceil(n * pct / 100)
    let num = n * pct;
    (num + 99) / 100
}

fn main() -> std::io::Result<()> {
    // Defaults match other benches.
    let mut nsp_list = vec![20usize, 40, 60];
    let mut tsp_list: Option<Vec<usize>> = None;
    let mut tsp_pct_list: Option<Vec<usize>> = Some(vec![50]);

    let mut kind = "all".to_string();
    let mut net = "all".to_string();
    let mut out_path = "augsso_bench.txt".to_string();

    let mut warmup = 50usize;
    let mut samples = 200usize;

    let mut rng_in_timed = false;

    let mut proc_warmup = 50usize;
    let mut proc_samples = 200usize;

    // Network model defaults 
    let mut lan_rtt_ms: f64 = 1.0;
    let mut lan_jitter_ms: f64 = 0.5;
    let mut lan_bw_mbps: f64 = 1000.0;

    let mut wan_rtt_ms: f64 = 50.0;
    let mut wan_jitter_ms: f64 = 5.0;
    let mut wan_bw_mbps: f64 = 50.0;

    let mut overhead_bytes: usize = 40;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--kind" => {
                i += 1;
                kind = args[i].clone();
            }
            "--net" => {
                i += 1;
                net = args[i].clone();
            }
            "--nsp" => {
                i += 1;
                nsp_list = parse_usize_list(&args[i]);
            }
            "--tsp" => {
                i += 1;
                tsp_list = Some(parse_usize_list(&args[i]));
                tsp_pct_list = None;
            }
            "--tsp-pct" => {
                i += 1;
                tsp_pct_list = Some(parse_usize_list(&args[i]));
                tsp_list = None;
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse().expect("bad warmup");
            }
            "--warmup-iters" => {
                i += 1;
                warmup = args[i].parse().expect("bad warmup-iters");
            }
            "--samples" => {
                i += 1;
                samples = args[i].parse().expect("bad samples");
            }
            "--sample-size" => {
                i += 1;
                samples = args[i].parse().expect("bad sample-size");
            }
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            "--rng-in-timed" => {
                rng_in_timed = true;
            }
            "--proc-warmup" => {
                i += 1;
                proc_warmup = args[i].parse().expect("bad proc_warmup");
            }
            "--proc-samples" => {
                i += 1;
                proc_samples = args[i].parse().expect("bad proc_samples");
            }
            "--lan-rtt-ms" => {
                i += 1;
                lan_rtt_ms = args[i].parse().expect("bad lan-rtt-ms");
            }
            "--lan-jitter-ms" => {
                i += 1;
                lan_jitter_ms = args[i].parse().expect("bad lan-jitter-ms");
            }
            "--lan-bw-mbps" => {
                i += 1;
                lan_bw_mbps = args[i].parse().expect("bad lan-bw-mbps");
            }
            "--wan-rtt-ms" => {
                i += 1;
                wan_rtt_ms = args[i].parse().expect("bad wan-rtt-ms");
            }
            "--wan-jitter-ms" => {
                i += 1;
                wan_jitter_ms = args[i].parse().expect("bad wan-jitter-ms");
            }
            "--wan-bw-mbps" => {
                i += 1;
                wan_bw_mbps = args[i].parse().expect("bad wan-bw-mbps");
            }
            "--overhead-bytes" => {
                i += 1;
                overhead_bytes = args[i].parse().expect("bad overhead-bytes");
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: bench_augsso [--kind proto|sp|prim|net|full|all] [--net lan|wan|all]\n\
                     \t[--nsp 20,40,60] [--tsp 10,20] | [--tsp-pct 50]\n\
                     \t[--warmup-iters N | --warmup N] [--sample-size N | --samples N]\n\
                     \t[--proc-warmup N] [--proc-samples N]\n\
                     \t[--lan-rtt-ms f] [--lan-jitter-ms f] [--lan-bw-mbps f]\n\
                     \t[--wan-rtt-ms f] [--wan-jitter-ms f] [--wan-bw-mbps f]\n\
                     \t[--overhead-bytes N]\n\
                     \t[--rng-in-timed] [--out path]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let f = File::create(&out_path)?;
    let mut out = BufWriter::new(f);
    write_header(&mut out)?;

    let lan = mk_profile("lan", lan_rtt_ms, lan_jitter_ms, lan_bw_mbps, overhead_bytes);
    let wan = mk_profile("wan", wan_rtt_ms, wan_jitter_ms, wan_bw_mbps, overhead_bytes);

    for &nsp in &nsp_list {
        if let Some(tsps) = tsp_list.as_ref() {
            for &tsp in tsps {
                run_one(
                    nsp,
                    tsp,
                    &kind,
                    &net,
                    warmup,
                    samples,
                    rng_in_timed,
                    proc_warmup,
                    proc_samples,
                    lan,
                    wan,
                    &mut out,
                )?;
            }
        } else {
            // tsp_pct
            let pcts: Vec<usize> = tsp_pct_list.clone().unwrap_or_else(|| vec![50]);
            for &pct in &pcts {
                let tsp = ceil_pct(nsp, pct);
                run_one(
                    nsp,
                    tsp,
                    &kind,
                    &net,
                    warmup,
                    samples,
                    rng_in_timed,
                    proc_warmup,
                    proc_samples,
                    lan,
                    wan,
                    &mut out,
                )?;
            }
        }
    }

    out.flush()?;
    Ok(())
}

fn run_one(
    nsp: usize,
    tsp: usize,
    kind: &str,
    net: &str,
    warmup: usize,
    samples: usize,
    rng_in_timed: bool,
    proc_warmup: usize,
    proc_samples: usize,
    lan: NetProfile,
    wan: NetProfile,
    out: &mut BufWriter<File>,
) -> std::io::Result<()> {
    let do_proto = kind == "proto" || kind == "all";
    let do_sp = kind == "sp" || kind == "all";
    let do_prim = kind == "prim" || kind == "all";
    let do_net = kind == "net" || kind == "all";
    let do_full = kind == "full" || kind == "all";

    if do_proto {
        bench_client_proto(nsp, tsp, warmup, samples, rng_in_timed, out)?;
    }
    if do_sp {
        bench_server_phases(nsp, tsp, warmup, samples, rng_in_timed, out)?;
    }
    if do_prim {
        bench_primitives(nsp, tsp, warmup, samples, rng_in_timed, out)?;
    }
    if do_net {
        match net {
            "lan" => bench_net(nsp, tsp, warmup, samples, lan, out)?,
            "wan" => bench_net(nsp, tsp, warmup, samples, wan, out)?,
            "all" => {
                bench_net(nsp, tsp, warmup, samples, lan, out)?;
                bench_net(nsp, tsp, warmup, samples, wan, out)?;
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "bad --net",
                ));
            }
        }
    }
    if do_full {
        match net {
            "lan" => bench_full(
                nsp,
                tsp,
                warmup,
                samples,
                rng_in_timed,
                lan,
                proc_warmup,
                proc_samples,
                out,
            )?,
            "wan" => bench_full(
                nsp,
                tsp,
                warmup,
                samples,
                rng_in_timed,
                wan,
                proc_warmup,
                proc_samples,
                out,
            )?,
            "all" => {
                bench_full(
                    nsp,
                    tsp,
                    warmup,
                    samples,
                    rng_in_timed,
                    lan,
                    proc_warmup,
                    proc_samples,
                    out,
                )?;
                bench_full(
                    nsp,
                    tsp,
                    warmup,
                    samples,
                    rng_in_timed,
                    wan,
                    proc_warmup,
                    proc_samples,
                    out,
                )?;
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "bad --net",
                ));
            }
        }
    }

    Ok(())
}
