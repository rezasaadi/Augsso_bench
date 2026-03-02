#![allow(clippy::needless_range_loop)]
//! - Setup uses dealer-style Shamir sharing for master secrets gamma and phai.
//! - Authentication uses pairing-based checks and reconstructs σ from t partials.
//! - Token issuance is a threshold BLS-style signature share on H(pld).
//! - PKE is a benchmark-friendly CPA-secure construction (X25519 + XChaCha20-Poly1305).

use std::collections::HashMap;

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use bls12_381::{G1Projective, G2Projective, Scalar as BScalar};
use ff::Field;

use crate::crypto_augsso as ac;
use crate::crypto_core;

pub const UID_LEN: usize = 32;
pub const X_LEN: usize = 32;

/// We model `pld` as `x || idU` 
pub const PLD_LEN: usize = X_LEN + UID_LEN; 

pub const BPW_LEN: usize = ac::G1_LEN; 
pub const SIGMA_LEN: usize = ac::G1_LEN; 

pub const CH_LEN: usize = 32;
pub const EK_LEN: usize = 32;
pub const PKE_PT_LEN: usize = CH_LEN + EK_LEN; 

pub const CTCH_PT_LEN: usize = CH_LEN; 
pub const CTTKN_PT_LEN: usize = ac::G1_LEN + PLD_LEN; 

pub const CTCH_LEN: usize = crypto_core::NONCE_LEN + CTCH_PT_LEN + crypto_core::TAG_LEN;
pub const CTTKN_LEN: usize = crypto_core::NONCE_LEN + CTTKN_PT_LEN + crypto_core::TAG_LEN;
pub const CTRES_LEN: usize = ac::PKE_EPHEMERAL_PK_LEN + crypto_core::NONCE_LEN + PKE_PT_LEN + crypto_core::TAG_LEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClientId(pub [u8; UID_LEN]);

#[derive(Clone, Copy, Debug)]
pub struct PublicParams {
    pub n: usize,
    pub t: usize,
    pub cnt: u64,
    pub caf: u64,
}


// Setup 

#[derive(Clone, Debug)]
pub struct SetupOut {
    pub pp: PublicParams,

    pub gamma: BScalar,
    pub phi: BScalar,

    pub gamma_shares: Vec<(u32, BScalar)>,
    pub phi_shares: Vec<(u32, BScalar)>,

    pub gamma_pub_shares: Vec<(u32, G2Projective)>,
    pub phi_pub_shares: Vec<(u32, G2Projective)>,

    pub gamma_pk: G2Projective,
    pub phi_pk: G2Projective,
}

pub fn setup(n: usize, t: usize, rng: &mut impl RngCore) -> SetupOut {
    let pp = PublicParams {
        n,
        t,
        cnt: u64::MAX / 2,
        caf: u64::MAX / 2,
    };

    let gamma = BScalar::random(&mut *rng);
    let phi = BScalar::random(&mut *rng);

    let gamma_shares = ac::shamir_share(gamma, n, t, rng);
    let phi_shares = ac::shamir_share(phi, n, t, rng);

    let g2 = G2Projective::generator();

    let gamma_pub_shares: Vec<(u32, G2Projective)> = gamma_shares
        .iter()
        .map(|(id, s)| (*id, g2 * *s))
        .collect();
    let phi_pub_shares: Vec<(u32, G2Projective)> = phi_shares
        .iter()
        .map(|(id, s)| (*id, g2 * *s))
        .collect();

    let gamma_pk = g2 * gamma;
    let phi_pk = g2 * phi;

    SetupOut {
        pp,
        gamma,
        phi,
        gamma_shares,
        phi_shares,
        gamma_pub_shares,
        phi_pub_shares,
        gamma_pk,
        phi_pk,
    }
}

// Registration

#[derive(Clone, Debug)]
pub struct RegistrationMsg {
    pub server_id: u32,
    pub sk_i: BScalar,
    pub rpk: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct RegistrationOut {
    pub sk: BScalar,
    pub rpk: [u8; 32],
    pub rsk: [u8; 32],
    pub msgs: Vec<RegistrationMsg>,
    pub pk_shares: Vec<(u32, G2Projective)>,
}

/// Client-side registration.
/// - Chooses `sk` and Shamir-splits it into `sk_i`.
/// - Computes hardened password `hpw = h(sk * H(pw) || pw)`.
/// - Derives deterministic `(rpk,rsk) = PKE.KG(hpw)`.
/// - Returns per-server messages `(sk_i, rpk)`.
pub fn registration(n: usize, t: usize, pw: &[u8], rng: &mut impl RngCore) -> RegistrationOut {
    let sk = ac::random_scalar_nonzero(rng);

    let h_pw = ac::hash_to_g1(b"augsso/H_pw/v1", pw);
    let sk_hpw = h_pw * sk;
    let hpw = ac::hash_g1_and_bytes_to_32(b"augsso/hpw/v1", &sk_hpw, pw);

    let kp = ac::pke_kg(&hpw);

    let shares = ac::shamir_share(sk, n, t, rng);
    let g2 = G2Projective::generator();

    let mut msgs = Vec::with_capacity(n);
    let mut pk_shares = Vec::with_capacity(n);
    for (sid, sk_i) in shares.iter() {
        msgs.push(RegistrationMsg {
            server_id: *sid,
            sk_i: *sk_i,
            rpk: kp.pk,
        });
        pk_shares.push((*sid, g2 * *sk_i));
    }

    RegistrationOut {
        sk,
        rpk: kp.pk,
        rsk: kp.sk,
        msgs,
        pk_shares,
    }
}

// Server state

#[derive(Clone, Debug)]
pub struct ServerRecord {
    pub sk_i: BScalar,
    pub rpk: [u8; 32],
    pub pk_i: G2Projective,
}

#[derive(Clone, Debug)]
pub struct AugSsoServer {
    pub id: u32,

    pub gamma_i: BScalar,
    pub phi_i: BScalar,

    pub gamma_pub_i: G2Projective,
    pub phi_pub_i: G2Projective,

    pub pp: PublicParams,

    records: HashMap<ClientId, ServerRecord>,
}

impl AugSsoServer {
    pub fn new(
        id: u32,
        gamma_i: BScalar,
        phi_i: BScalar,
        gamma_pub_i: G2Projective,
        phi_pub_i: G2Projective,
        pp: PublicParams,
    ) -> Self {
        Self {
            id,
            gamma_i,
            phi_i,
            gamma_pub_i,
            phi_pub_i,
            pp,
            records: HashMap::new(),
        }
    }

    pub fn store(&mut self, c: ClientId, msg: &RegistrationMsg) {
        debug_assert_eq!(self.id, msg.server_id);
        let pk_i = G2Projective::generator() * msg.sk_i;
        self.records.insert(
            c,
            ServerRecord {
                sk_i: msg.sk_i,
                rpk: msg.rpk,
                pk_i,
            },
        );
    }

    pub fn has_record(&self, c: ClientId) -> bool {
        self.records.contains_key(&c)
    }

    pub fn get_record(&self, c: ClientId) -> Option<&ServerRecord> {
        self.records.get(&c)
    }
}

// Authentication + token issuance

#[derive(Clone, Debug)]
pub struct ClientState {
    pub password: Vec<u8>,
    pub r: BScalar,
    pub t_set: Vec<u32>,
    pub pld: [u8; PLD_LEN],
}

#[derive(Clone, Debug)]
pub struct ClientRequest {
    pub pld: [u8; PLD_LEN],
    pub bpw: [u8; BPW_LEN],
}

pub fn request(
    password: &[u8],
    pld: [u8; PLD_LEN],
    t_set: &[u32],
    rng: &mut impl RngCore,
) -> (ClientState, ClientRequest) {
    let r = ac::random_scalar_nonzero(rng);
    request_with_r(password, pld, t_set, r)
}

pub fn request_with_r(
    password: &[u8],
    pld: [u8; PLD_LEN],
    t_set: &[u32],
    r: BScalar,
) -> (ClientState, ClientRequest) {
    let h_pw = ac::hash_to_g1(b"augsso/H_pw/v1", password);
    let bpw_pt = h_pw * r;
    let bpw = ac::g1_to_bytes(&bpw_pt);

    let st = ClientState {
        password: password.to_vec(),
        r,
        t_set: t_set.to_vec(),
        pld,
    };

    let req = ClientRequest { pld, bpw };
    (st, req)
}

/// Server-side randomness for Respond1 
#[derive(Clone, Debug)]
pub struct Respond1Rand {
    pub ch: [u8; CH_LEN],
    pub ek: [u8; EK_LEN],
    pub pke_eph_sk: [u8; 32],
    pub pke_nonce: [u8; crypto_core::NONCE_LEN],
}

#[derive(Clone, Debug)]
pub struct ServerResponse1 {
    pub server_id: u32,
    pub sigma_i: [u8; SIGMA_LEN],
    pub ct_res: ac::PkeCt<PKE_PT_LEN>,
}

#[derive(Clone, Debug)]
pub struct ServerSession {
    pub server_id: u32,
    pub ek: [u8; EK_LEN],
    pub ch: [u8; CH_LEN],
    pub pld: [u8; PLD_LEN],
}

fn parse_client_id_from_pld(pld: &[u8; PLD_LEN]) -> ClientId {
    let mut id = [0u8; UID_LEN];
    id.copy_from_slice(&pld[X_LEN..X_LEN + UID_LEN]);
    ClientId(id)
}

pub fn respond1(
    srv: &AugSsoServer,
    req: &ClientRequest,
    rng: &mut impl RngCore,
) -> Option<(ServerResponse1, ServerSession)> {
    // Require a record.
    let c = parse_client_id_from_pld(&req.pld);
    if !srv.records.contains_key(&c) {
        return None;
    }

    // sample ek and ch
    let mut ek = [0u8; EK_LEN];
    rng.fill_bytes(&mut ek);
    let mut ch = [0u8; CH_LEN];
    rng.fill_bytes(&mut ch);

    let mut pke_eph_sk = [0u8; 32];
    rng.fill_bytes(&mut pke_eph_sk);
    let mut pke_nonce = [0u8; crypto_core::NONCE_LEN];
    rng.fill_bytes(&mut pke_nonce);

    let rand = Respond1Rand {
        ch,
        ek,
        pke_eph_sk,
        pke_nonce,
    };

    respond1_with_rand(srv, req, &rand)
}

pub fn respond1_with_rand(
    srv: &AugSsoServer,
    req: &ClientRequest,
    rand: &Respond1Rand,
) -> Option<(ServerResponse1, ServerSession)> {
    let c = parse_client_id_from_pld(&req.pld);
    let rec = srv.records.get(&c)?;

    let bpw_pt = ac::g1_from_bytes(&req.bpw)?;
    let sigma_pt = bpw_pt * rec.sk_i;
    let sigma_i = ac::g1_to_bytes(&sigma_pt);
    let mut pt = [0u8; PKE_PT_LEN];
    pt[0..CH_LEN].copy_from_slice(&rand.ch);
    pt[CH_LEN..CH_LEN + EK_LEN].copy_from_slice(&rand.ek);

    let aad: [u8; 0] = [];
    let ct_res = ac::pke_enc_with_eph_and_nonce::<PKE_PT_LEN>(
        &rec.rpk,
        &aad,
        &pt,
        &rand.pke_eph_sk,
        &rand.pke_nonce,
    );

    let resp = ServerResponse1 {
        server_id: srv.id,
        sigma_i,
        ct_res,
    };

    let sess = ServerSession {
        server_id: srv.id,
        ek: rand.ek,
        ch: rand.ch,
        pld: req.pld,
    };

    Some((resp, sess))
}

#[derive(Clone, Debug)]
pub struct ClientPhase1Out {
    pub ek: Vec<(u32, [u8; EK_LEN])>,
    pub ch: Vec<(u32, [u8; CH_LEN])>,
    pub ct_ch: Vec<(u32, crypto_core::CtBlob<CTCH_PT_LEN>)>,
}

/// Client processes ServerResponse1 messages:
/// - verifies σ_i (pairing)
/// - reconstructs σ and derives rsk
/// - decrypts CTres_i to obtain (ch_i,ek_i)
/// - returns CTch_i = SE.Enc(ek_i, ch_i)
/// `ctch_nonces` must be provided (one per server in `t_set`) when running in deterministic mode.
pub fn client_phase1(
    st: &ClientState,
    req: &ClientRequest,
    resps: &[ServerResponse1],
    pk_shares: &[(u32, G2Projective)],
    ctch_nonces: Option<&[[u8; crypto_core::NONCE_LEN]]>,
    rng: &mut impl RngCore,
) -> Option<ClientPhase1Out> {
    if resps.len() != st.t_set.len() {
        return None;
    }

    // Ensure server id set matches.
    let mut ids: Vec<u32> = resps.iter().map(|r| r.server_id).collect();
    ids.sort_unstable();
    let mut t_sorted = st.t_set.clone();
    t_sorted.sort_unstable();
    if ids != t_sorted {
        return None;
    }

    let bpw_pt = ac::g1_from_bytes(&req.bpw)?;
    for &sid in &t_sorted {
        let r = resps.iter().find(|r| r.server_id == sid).unwrap();
        let sigma_pt = ac::g1_from_bytes(&r.sigma_i)?;
        let pk_i = pk_shares.iter().find(|(id, _)| *id == sid)?.1;
        if !ac::pairing_check_g1_g2(&sigma_pt, &bpw_pt, &pk_i) {
            return None;
        }
    }

    // Reconstruct sigma = r^{-1} * Σ λ_i σ_i
    let mut partials = Vec::with_capacity(t_sorted.len());
    for &sid in &t_sorted {
        let r = resps.iter().find(|r| r.server_id == sid).unwrap();
        partials.push(ac::g1_from_bytes(&r.sigma_i)?);
    }
    let acc = ac::combine_g1_at_zero(&t_sorted, &partials);
    let rinv = st.r.invert();
    let rinv = Option::<BScalar>::from(rinv).unwrap_or(BScalar::zero());
    let sigma = acc * rinv;

    // hpw = h(sigma || pw)
    // IMPORTANT: this must match the domain separator used during registration,
    // otherwise the derived (rpk,rsk) will differ and PKE decryption will fail.
    let hpw = ac::hash_g1_and_bytes_to_32(b"augsso/hpw/v1", &sigma, &st.password);
    let kp = ac::pke_kg(&hpw);

    // Decrypt each CTres_i to get (ch_i, ek_i)
    let aad: [u8; 0] = [];
    let mut ek_out: Vec<(u32, [u8; EK_LEN])> = Vec::with_capacity(t_sorted.len());
    let mut ch_out: Vec<(u32, [u8; CH_LEN])> = Vec::with_capacity(t_sorted.len());

    for &sid in &t_sorted {
        let r = resps.iter().find(|r| r.server_id == sid).unwrap();
        let pt = ac::pke_dec::<PKE_PT_LEN>(&kp.sk, &aad, &r.ct_res).ok()?;
        let mut ch = [0u8; CH_LEN];
        let mut ek = [0u8; EK_LEN];
        ch.copy_from_slice(&pt[0..CH_LEN]);
        ek.copy_from_slice(&pt[CH_LEN..CH_LEN + EK_LEN]);
        ch_out.push((sid, ch));
        ek_out.push((sid, ek));
    }

    // Build CTch_i = SE.Enc(ek_i, ch_i)
    let mut ct_ch: Vec<(u32, crypto_core::CtBlob<CTCH_PT_LEN>)> = Vec::with_capacity(t_sorted.len());
    let aad2: [u8; 0] = [];

    for (idx, &sid) in t_sorted.iter().enumerate() {
        let ek = ek_out.iter().find(|(id, _)| *id == sid).unwrap().1;
        let ch = ch_out.iter().find(|(id, _)| *id == sid).unwrap().1;
        let blob = if let Some(ns) = ctch_nonces {
            ac::xchacha_encrypt_detached_with_nonce::<CTCH_PT_LEN>(&ek, &aad2, &ch, &ns[idx])
        } else {
            ac::xchacha_encrypt_detached::<CTCH_PT_LEN>(&ek, &aad2, &ch, rng)
        };
        ct_ch.push((sid, blob));
    }

    Some(ClientPhase1Out { ek: ek_out, ch: ch_out, ct_ch })
}

#[derive(Clone, Debug)]
pub struct ServerResponse2 {
    pub server_id: u32,
    pub ct_tkn: crypto_core::CtBlob<CTTKN_PT_LEN>,
}

pub fn respond2(
    srv: &AugSsoServer,
    sess: &ServerSession,
    ct_ch: &crypto_core::CtBlob<CTCH_PT_LEN>,
    rng: &mut impl RngCore,
) -> Option<ServerResponse2> {
    let mut nonce = [0u8; crypto_core::NONCE_LEN];
    rng.fill_bytes(&mut nonce);
    respond2_with_nonce(srv, sess, ct_ch, &nonce)
}

pub fn respond2_with_nonce(
    srv: &AugSsoServer,
    sess: &ServerSession,
    ct_ch: &crypto_core::CtBlob<CTCH_PT_LEN>,
    nonce: &[u8; crypto_core::NONCE_LEN],
) -> Option<ServerResponse2> {
    // decrypt CTch and compare
    let aad: [u8; 0] = [];
    let ch_recv = ac::xchacha_decrypt_detached::<CTCH_PT_LEN>(&sess.ek, &aad, ct_ch).ok()?;
    if ch_recv != sess.ch {
        return None;
    }

    // token share tkn_i = gamma_i * H(pld)
    let h_pld = ac::hash_to_g1(b"augsso/H_pld/v1", &sess.pld);
    let tkn_i = h_pld * srv.gamma_i;

    let mut pt = [0u8; CTTKN_PT_LEN];
    let tkn_bytes = ac::g1_to_bytes(&tkn_i);
    pt[0..ac::G1_LEN].copy_from_slice(&tkn_bytes);
    pt[ac::G1_LEN..ac::G1_LEN + PLD_LEN].copy_from_slice(&sess.pld);

    let blob = ac::xchacha_encrypt_detached_with_nonce::<CTTKN_PT_LEN>(&sess.ek, &aad, &pt, nonce);

    Some(ServerResponse2 {
        server_id: srv.id,
        ct_tkn: blob,
    })
}

#[derive(Clone, Debug)]
pub struct Token {
    pub pld: [u8; PLD_LEN],
    pub tkn: [u8; ac::G1_LEN],
}

pub fn client_finalize(
    st: &ClientState,
    phase1: &ClientPhase1Out,
    resps2: &[ServerResponse2],
    gamma_pub_shares: &[(u32, G2Projective)],
) -> Option<Token> {
    if resps2.len() != st.t_set.len() {
        return None;
    }

    // Ensure server id set matches.
    let mut ids: Vec<u32> = resps2.iter().map(|r| r.server_id).collect();
    ids.sort_unstable();
    let mut t_sorted = st.t_set.clone();
    t_sorted.sort_unstable();
    if ids != t_sorted {
        return None;
    }

    // Build ek lookup.
    let mut ek_map: HashMap<u32, [u8; EK_LEN]> = HashMap::new();
    for (sid, ek) in &phase1.ek {
        ek_map.insert(*sid, *ek);
    }

    let aad: [u8; 0] = [];
    let h_pld = ac::hash_to_g1(b"augsso/H_pld/v1", &st.pld);

    let mut partials: Vec<G1Projective> = Vec::with_capacity(t_sorted.len());

    for &sid in &t_sorted {
        let r2 = resps2.iter().find(|r| r.server_id == sid).unwrap();
        let ek = ek_map.get(&sid)?;
        let pt = ac::xchacha_decrypt_detached::<CTTKN_PT_LEN>(ek, &aad, &r2.ct_tkn).ok()?;

        let mut tkn_bytes = [0u8; ac::G1_LEN];
        tkn_bytes.copy_from_slice(&pt[0..ac::G1_LEN]);
        let mut pld_bytes = [0u8; PLD_LEN];
        pld_bytes.copy_from_slice(&pt[ac::G1_LEN..ac::G1_LEN + PLD_LEN]);
        if pld_bytes != st.pld {
            return None;
        }

        let tkn_i = ac::g1_from_bytes(&tkn_bytes)?;

        // verify partial token: e(tkn_i, g2) == e(H(pld), Gamma_i)
        let gamma_pub_i = gamma_pub_shares.iter().find(|(id, _)| *id == sid)?.1;
        if !ac::pairing_check_sig(&h_pld, &tkn_i, &gamma_pub_i) {
            return None;
        }

        partials.push(tkn_i);
    }

    let tkn = ac::combine_g1_at_zero(&t_sorted, &partials);
    let tkn_bytes = ac::g1_to_bytes(&tkn);

    Some(Token {
        pld: st.pld,
        tkn: tkn_bytes,
    })
}

pub fn verify_token(gamma_pk: &G2Projective, token: &Token) -> bool {
    let h_pld = ac::hash_to_g1(b"augsso/H_pld/v1", &token.pld);
    let sig = match ac::g1_from_bytes(&token.tkn) {
        Some(s) => s,
        None => return false,
    };
    ac::pairing_check_sig(&h_pld, &sig, gamma_pk)
}

// Fixtures for benchmarking

#[derive(Clone)]
pub struct Fixture {
    pub n: usize,
    pub t: usize,

    pub c: ClientId,
    pub x: [u8; X_LEN],
    pub pld: [u8; PLD_LEN],
    pub password: Vec<u8>,

    pub gamma_pk: G2Projective,
    pub gamma_pub_shares: Vec<(u32, G2Projective)>,

    pub pk_shares: Vec<(u32, G2Projective)>,

    pub servers: Vec<AugSsoServer>,
    pub t_set: Vec<u32>,
}

fn seed_for(tag: &[u8], n: usize, t: usize) -> [u8; 32] {
    use blake3;
    let mut h = blake3::Hasher::new();
    h.update(tag);
    h.update(&(n as u64).to_le_bytes());
    h.update(&(t as u64).to_le_bytes());
    let out = h.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(out.as_bytes());
    seed
}

pub fn make_fixture(n: usize, t: usize) -> Fixture {
    assert!(t >= 1 && t <= n);

    let mut rng = ChaCha20Rng::from_seed(seed_for(b"augsso/fixture/v1", n, t));

    let mut c_bytes = [0u8; UID_LEN];
    rng.fill_bytes(&mut c_bytes);
    let c = ClientId(c_bytes);

    let mut x = [0u8; X_LEN];
    rng.fill_bytes(&mut x);

    // pld = x || id
    let mut pld = [0u8; PLD_LEN];
    pld[0..X_LEN].copy_from_slice(&x);
    pld[X_LEN..X_LEN + UID_LEN].copy_from_slice(&c.0);

    let password = b"correct horse battery staple".to_vec();

    // Setup master secrets and create servers.
    let s = setup(n, t, &mut rng);

    let mut servers: Vec<AugSsoServer> = Vec::with_capacity(n);
    for i in 0..n {
        let sid = (i + 1) as u32;
        let gamma_i = s.gamma_shares[i].1;
        let phi_i = s.phi_shares[i].1;
        let gamma_pub_i = s.gamma_pub_shares[i].1;
        let phi_pub_i = s.phi_pub_shares[i].1;
        servers.push(AugSsoServer::new(
            sid,
            gamma_i,
            phi_i,
            gamma_pub_i,
            phi_pub_i,
            s.pp,
        ));
    }
    servers.sort_by_key(|s| s.id);

    // Registration and Store.
    let reg = registration(n, t, &password, &mut rng);
    for msg in reg.msgs.iter() {
        let idx = (msg.server_id - 1) as usize;
        servers[idx].store(c, msg);
    }

    let t_set: Vec<u32> = (1..=t as u32).collect();

    Fixture {
        n,
        t,
        c,
        x,
        pld,
        password,
        gamma_pk: s.gamma_pk,
        gamma_pub_shares: s.gamma_pub_shares,
        pk_shares: reg.pk_shares,
        servers,
        t_set,
    }
}

#[derive(Clone)]
pub struct IterData {
    pub r: BScalar,
    pub ctch_nonces: Vec<[u8; crypto_core::NONCE_LEN]>,
    pub cttkn_nonces: Vec<[u8; crypto_core::NONCE_LEN]>,
    pub r1: Vec<Respond1Rand>,
}

pub fn make_iter_data(fx: &Fixture, rng: &mut impl RngCore) -> IterData {
    let r = ac::random_scalar_nonzero(rng);

    let mut ctch_nonces = vec![[0u8; crypto_core::NONCE_LEN]; fx.t];
    for n in ctch_nonces.iter_mut() {
        rng.fill_bytes(n);
    }

    let mut cttkn_nonces = vec![[0u8; crypto_core::NONCE_LEN]; fx.t];
    for n in cttkn_nonces.iter_mut() {
        rng.fill_bytes(n);
    }

    let mut r1 = Vec::with_capacity(fx.t);
    for _ in 0..fx.t {
        let mut ch = [0u8; CH_LEN];
        let mut ek = [0u8; EK_LEN];
        let mut eph = [0u8; 32];
        let mut nonce = [0u8; crypto_core::NONCE_LEN];
        rng.fill_bytes(&mut ch);
        rng.fill_bytes(&mut ek);
        rng.fill_bytes(&mut eph);
        rng.fill_bytes(&mut nonce);
        r1.push(Respond1Rand {
            ch,
            ek,
            pke_eph_sk: eph,
            pke_nonce: nonce,
        });
    }

    IterData {
        r,
        ctch_nonces,
        cttkn_nonces,
        r1,
    }
}

// Tests (basic sanity)

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn roundtrip_auth_and_verify_token() {
        // Small parameters keep this fast while still exercising threshold logic.
        let n = 5;
        let t = 3;
        let fx = make_fixture(n, t);

        let mut rng_it = ChaCha20Rng::from_seed(super::seed_for(b"augsso/test/it", n, t));
        let mut rng_cli = ChaCha20Rng::from_seed(super::seed_for(b"augsso/test/cli", n, t));

        let it = make_iter_data(&fx, &mut rng_it);
        let (st, req) = request_with_r(&fx.password, fx.pld, &fx.t_set, it.r);

        // Server phase 1
        let mut resps1 = Vec::with_capacity(t);
        let mut sess = Vec::with_capacity(t);
        for j in 0..t {
            let (r1, s1) = respond1_with_rand(&fx.servers[j], &req, &it.r1[j])
                .expect("respond1");
            resps1.push(r1);
            sess.push(s1);
        }

        // Client phase 1
        let phase1 = client_phase1(
            &st,
            &req,
            &resps1,
            &fx.pk_shares,
            Some(&it.ctch_nonces),
            &mut rng_cli,
        )
        .expect("client_phase1");

        // Server phase 2
        let mut resps2 = Vec::with_capacity(t);
        for j in 0..t {
            let sid = (j + 1) as u32;
            let ctch = phase1
                .ct_ch
                .iter()
                .find(|(id, _)| *id == sid)
                .expect("ctch")
                .1
                .clone();
            let r2 = respond2_with_nonce(&fx.servers[j], &sess[j], &ctch, &it.cttkn_nonces[j])
                .expect("respond2");
            resps2.push(r2);
        }

        // Finalize + verify
        let tok = client_finalize(&st, &phase1, &resps2, &fx.gamma_pub_shares)
            .expect("client_finalize");
        assert!(verify_token(&fx.gamma_pk, &tok));
    }
}

