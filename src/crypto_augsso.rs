// - Pairing-friendly group operations (BLS12-381)
// - Hash-to-group and hash-to-scalar (benchmark-friendly, not IETF hash-to-curve)
// - Shamir secret sharing helpers in the BLS scalar field
// - A simple CPA-secure PKE for the benchmark harness: X25519 + XChaCha20-Poly1305

use bls12_381::{pairing, G1Affine, G1Projective, G2Affine, G2Projective, Scalar as BScalar};
use ff::Field;
use rand_core::RngCore;
use x25519_dalek::{PublicKey as X25519Pk, StaticSecret as X25519Sk};

use crate::crypto_core;

pub const G1_LEN: usize = 48;

pub type Scalar = BScalar;
pub type G1 = G1Projective;
pub type G2 = G2Projective;

// Hashes (benchmark-friendly)

/// H : {0,1}* -> G1
pub fn hash_to_g1(domain: &[u8], msg: &[u8]) -> G1 {
    let wide = crypto_core::blake3_xof_64(domain, &[msg]);
    let s = Scalar::from_bytes_wide(&wide);
    G1::generator() * s
}

/// h : {0,1}* -> Zp* (non-zero scalar)
pub fn hash_to_scalar_nonzero(domain: &[u8], msg: &[u8]) -> Scalar {
    let wide = crypto_core::blake3_xof_64(domain, &[msg]);
    let s = Scalar::from_bytes_wide(&wide);
    if s == Scalar::zero() {
        Scalar::ONE
    } else {
        s
    }
}

/// h' : Zp* -> {0,1}^λ (we use λ=256 bits)
pub fn hash_scalar_to_32(domain: &[u8], s: &Scalar) -> [u8; 32] {
    crypto_core::blake3_32(domain, &[&s.to_bytes()])
}

/// h : (group element, bytes) -> {0,1}^λ
pub fn hash_g1_and_bytes_to_32(domain: &[u8], p: &G1, b: &[u8]) -> [u8; 32] {
    let p_bytes = G1Affine::from(*p).to_compressed();
    crypto_core::blake3_32(domain, &[&p_bytes, b])
}


// Shamir sharing 

pub fn random_scalar_nonzero(rng: &mut impl RngCore) -> Scalar {
    loop {
        let s = Scalar::random(&mut *rng);
        if s != Scalar::zero() {
            return s;
        }
    }
}

/// Dealer-style Shamir share generation for a secret `s`.
/// Returns (s, shares[(id, share)]).
pub fn shamir_share(secret: Scalar, n: usize, t: usize, rng: &mut impl RngCore) -> Vec<(u32, Scalar)> {
    assert!(t >= 1 && t <= n);
    let mut coeffs = Vec::with_capacity(t);
    coeffs.push(secret);
    for _ in 1..t {
        coeffs.push(Scalar::random(&mut *rng));
    }

    fn eval(coeffs: &[Scalar], x: Scalar) -> Scalar {
        let mut acc = Scalar::zero();
        let mut pow = Scalar::ONE;
        for c in coeffs {
            acc += *c * pow;
            pow *= x;
        }
        acc
    }

    let mut shares = Vec::with_capacity(n);
    for i in 1..=n {
        shares.push((i as u32, eval(&coeffs, Scalar::from(i as u64))));
    }
    shares
}

/// Lagrange coefficients at x=0 for the provided share ids.
pub fn lagrange_coeffs_at_zero(ids: &[u32]) -> Vec<Scalar> {
    let xs: Vec<Scalar> = ids.iter().map(|&i| Scalar::from(i as u64)).collect();
    let mut lambdas = Vec::with_capacity(xs.len());

    for i in 0..xs.len() {
        let mut num = Scalar::ONE;
        let mut den = Scalar::ONE;
        for j in 0..xs.len() {
            if i != j {
                num *= xs[j];
                den *= xs[j] - xs[i];
            }
        }
        let inv = den.invert();
        let inv = Option::<Scalar>::from(inv).unwrap_or(Scalar::zero());
        lambdas.push(num * inv);
    }

    lambdas
}

pub fn combine_g1_at_zero(ids: &[u32], partials: &[G1]) -> G1 {
    assert_eq!(ids.len(), partials.len());
    let lambdas = lagrange_coeffs_at_zero(ids);
    let mut acc = G1::identity();
    for (p, l) in partials.iter().zip(lambdas.iter()) {
        acc += *p * *l;
    }
    acc
}

pub fn combine_g2_at_zero(ids: &[u32], partials: &[G2]) -> G2 {
    assert_eq!(ids.len(), partials.len());
    let lambdas = lagrange_coeffs_at_zero(ids);
    let mut acc = G2::identity();
    for (p, l) in partials.iter().zip(lambdas.iter()) {
        acc += *p * *l;
    }
    acc
}

// Pairing checks

pub fn pairing_check_g1_g2(lhs_g1: &G1, rhs_g1: &G1, rhs_g2: &G2) -> bool {
    // e(lhs_g1, g2_gen) == e(rhs_g1, rhs_g2)
    let lhs = G1Affine::from(*lhs_g1);
    let g2 = G2Affine::from(G2::generator());
    let rhs1 = G1Affine::from(*rhs_g1);
    let rhs2 = G2Affine::from(*rhs_g2);
    pairing(&lhs, &g2) == pairing(&rhs1, &rhs2)
}

pub fn pairing_check_sig(h: &G1, sig: &G1, pk: &G2) -> bool {
    // e(sig, g2) == e(h, pk)
    let sig_a = G1Affine::from(*sig);
    let g2 = G2Affine::from(G2::generator());
    let h_a = G1Affine::from(*h);
    let pk_a = G2Affine::from(*pk);
    pairing(&sig_a, &g2) == pairing(&h_a, &pk_a)
}

// SE (symmetric encryption)

pub use crate::crypto_core::{
    xchacha_decrypt_detached, xchacha_encrypt_detached, xchacha_encrypt_detached_with_nonce, CtBlob,
    NONCE_LEN, TAG_LEN,
};

// PKE (benchmark-friendly, CPA-secure)

pub const PKE_PK_LEN: usize = 32;
pub const PKE_SK_LEN: usize = 32;
pub const PKE_EPHEMERAL_PK_LEN: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct PkeKeypair {
    pub pk: [u8; PKE_PK_LEN],
    pub sk: [u8; PKE_SK_LEN],
}

#[derive(Clone, Debug)]
pub struct PkeCt<const PT_LEN: usize> {
    pub eph_pk: [u8; PKE_EPHEMERAL_PK_LEN],
    pub blob: CtBlob<PT_LEN>,
}

pub fn pke_kg(seed32: &[u8; 32]) -> PkeKeypair {
    let sk_bytes = crypto_core::blake3_32(b"augsso/pke/kg/v1", &[seed32]);
    let sk = X25519Sk::from(sk_bytes);
    let pk = X25519Pk::from(&sk);

    PkeKeypair {
        pk: pk.to_bytes(),
        sk: sk.to_bytes(),
    }
}

fn pke_kdf(shared: &[u8; 32], eph_pk: &[u8; 32], pk: &[u8; 32]) -> [u8; 32] {
    crypto_core::blake3_32(b"augsso/pke/kdf/v1", &[shared, eph_pk, pk])
}

pub fn pke_enc<const PT_LEN: usize>(
    pk_bytes: &[u8; 32],
    aad: &[u8],
    plaintext: &[u8; PT_LEN],
    rng: &mut impl RngCore,
) -> PkeCt<PT_LEN> {
    let mut eph_sk_bytes = [0u8; 32];
    rng.fill_bytes(&mut eph_sk_bytes);
    let eph_sk = X25519Sk::from(eph_sk_bytes);
    let eph_pk = X25519Pk::from(&eph_sk);

    let pk = X25519Pk::from(*pk_bytes);
    let shared = eph_sk.diffie_hellman(&pk).to_bytes();
    let key = pke_kdf(&shared, &eph_pk.to_bytes(), pk_bytes);

    let blob = crypto_core::xchacha_encrypt_detached::<PT_LEN>(&key, aad, plaintext, rng);

    PkeCt {
        eph_pk: eph_pk.to_bytes(),
        blob,
    }
}

/// Deterministic PKE encryption 
pub fn pke_enc_with_eph_and_nonce<const PT_LEN: usize>(
    pk_bytes: &[u8; 32],
    aad: &[u8],
    plaintext: &[u8; PT_LEN],
    eph_sk_bytes: &[u8; 32],
    nonce: &[u8; crypto_core::NONCE_LEN],
) -> PkeCt<PT_LEN> {
    let eph_sk = X25519Sk::from(*eph_sk_bytes);
    let eph_pk = X25519Pk::from(&eph_sk);

    let pk = X25519Pk::from(*pk_bytes);
    let shared = eph_sk.diffie_hellman(&pk).to_bytes();
    let key = pke_kdf(&shared, &eph_pk.to_bytes(), pk_bytes);

    let blob = crypto_core::xchacha_encrypt_detached_with_nonce::<PT_LEN>(&key, aad, plaintext, nonce);

    PkeCt {
        eph_pk: eph_pk.to_bytes(),
        blob,
    }
}

pub fn pke_dec<const PT_LEN: usize>(
    sk_bytes: &[u8; 32],
    aad: &[u8],
    ct: &PkeCt<PT_LEN>,
) -> Result<[u8; PT_LEN], chacha20poly1305::aead::Error> {
    let sk = X25519Sk::from(*sk_bytes);
    let eph_pk = X25519Pk::from(ct.eph_pk);
    let shared = sk.diffie_hellman(&eph_pk).to_bytes();
    let pk = X25519Pk::from(&sk);
    let key = pke_kdf(&shared, &ct.eph_pk, &pk.to_bytes());

    crypto_core::xchacha_decrypt_detached::<PT_LEN>(&key, aad, &ct.blob)
}

pub fn pke_ct_len<const PT_LEN: usize>() -> usize {
    PKE_EPHEMERAL_PK_LEN + crypto_core::NONCE_LEN + PT_LEN + crypto_core::TAG_LEN
}

pub fn pke_serialize<const PT_LEN: usize>(ct: &PkeCt<PT_LEN>) -> Vec<u8> {
    let mut out = Vec::with_capacity(pke_ct_len::<PT_LEN>());
    out.extend_from_slice(&ct.eph_pk);
    out.extend_from_slice(&ct.blob.nonce);
    out.extend_from_slice(&ct.blob.ct);
    out.extend_from_slice(&ct.blob.tag);
    out
}

pub fn pke_deserialize<const PT_LEN: usize>(bytes: &[u8]) -> Option<PkeCt<PT_LEN>> {
    if bytes.len() != pke_ct_len::<PT_LEN>() {
        return None;
    }
    let mut off = 0;
    let mut eph_pk = [0u8; PKE_EPHEMERAL_PK_LEN];
    eph_pk.copy_from_slice(&bytes[off..off + PKE_EPHEMERAL_PK_LEN]);
    off += PKE_EPHEMERAL_PK_LEN;

    let mut nonce = [0u8; crypto_core::NONCE_LEN];
    nonce.copy_from_slice(&bytes[off..off + crypto_core::NONCE_LEN]);
    off += crypto_core::NONCE_LEN;

    let mut ct_bytes = [0u8; PT_LEN];
    ct_bytes.copy_from_slice(&bytes[off..off + PT_LEN]);
    off += PT_LEN;

    let mut tag = [0u8; crypto_core::TAG_LEN];
    tag.copy_from_slice(&bytes[off..off + crypto_core::TAG_LEN]);

    Some(PkeCt {
        eph_pk,
        blob: CtBlob { nonce, ct: ct_bytes, tag },
    })
}

// Convenience helpers for (de)serializing G1/G2

pub fn g1_to_bytes(p: &G1) -> [u8; G1_LEN] {
    G1Affine::from(*p).to_compressed()
}

pub fn g1_from_bytes(b: &[u8; G1_LEN]) -> Option<G1> {
    let aff = Option::<G1Affine>::from(G1Affine::from_compressed(b))?;
    Some(G1Projective::from(aff))
}

pub fn g2_to_bytes(p: &G2) -> [u8; G2_LEN] {
    G2Affine::from(*p).to_compressed()
}

pub fn g2_from_bytes(b: &[u8; G2_LEN]) -> Option<G2> {
    let aff = Option::<G2Affine>::from(G2Affine::from_compressed(b))?;
    Some(G2Projective::from(aff))
}

