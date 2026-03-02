
//This intentionally mirrors the layout in the provided PAS-TA-U / PASTA benches:
//- Blake3 for hashing / KDF
//- XChaCha20-Poly1305 for AEAD (used as `SE`)

use blake3;
use chacha20poly1305::{
    aead::{
        generic_array::GenericArray,
        AeadInPlace, Error as AeadError, KeyInit,
    },
    XChaCha20Poly1305, XNonce,
};
use rand_core::RngCore;

pub const NONCE_LEN: usize = 24;
pub const TAG_LEN: usize = 16;

#[derive(Clone, Debug)]
pub struct CtBlob<const PT_LEN: usize> {
    pub nonce: [u8; NONCE_LEN],
    pub ct: [u8; PT_LEN],
    pub tag: [u8; TAG_LEN],
}

pub fn blake3_32(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    for p in parts {
        h.update(p);
    }
    let out = h.finalize();
    let mut r = [0u8; 32];
    r.copy_from_slice(out.as_bytes());
    r
}

pub fn blake3_xof_64(domain: &[u8], parts: &[&[u8]]) -> [u8; 64] {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    for p in parts {
        h.update(p);
    }
    let mut wide = [0u8; 64];
    h.finalize_xof().fill(&mut wide);
    wide
}

pub fn xchacha_encrypt_detached<const PT_LEN: usize>(
    key: &[u8; 32],
    aad: &[u8],
    plaintext: &[u8; PT_LEN],
    rng: &mut impl RngCore,
) -> CtBlob<PT_LEN> {
    let cipher = XChaCha20Poly1305::new_from_slice(key).unwrap();

    let mut nonce = [0u8; NONCE_LEN];
    rng.fill_bytes(&mut nonce);
    let xnonce = XNonce::from_slice(&nonce);

    let mut ct = *plaintext;
    let tag = cipher.encrypt_in_place_detached(xnonce, aad, &mut ct).unwrap();

    let mut tag_bytes = [0u8; TAG_LEN];
    tag_bytes.copy_from_slice(tag.as_slice());

    CtBlob { nonce, ct, tag: tag_bytes }
}

/// Deterministic AEAD helper 
pub fn xchacha_encrypt_detached_with_nonce<const PT_LEN: usize>(
    key: &[u8; 32],
    aad: &[u8],
    plaintext: &[u8; PT_LEN],
    nonce: &[u8; NONCE_LEN],
) -> CtBlob<PT_LEN> {
    let cipher = XChaCha20Poly1305::new_from_slice(key).unwrap();
    let xnonce = XNonce::from_slice(nonce);

    let mut ct = *plaintext;
    let tag = cipher.encrypt_in_place_detached(xnonce, aad, &mut ct).unwrap();

    let mut tag_bytes = [0u8; TAG_LEN];
    tag_bytes.copy_from_slice(tag.as_slice());

    let mut nonce_bytes = [0u8; NONCE_LEN];
    nonce_bytes.copy_from_slice(nonce);

    CtBlob {
        nonce: nonce_bytes,
        ct,
        tag: tag_bytes,
    }
}

pub fn xchacha_decrypt_detached<const PT_LEN: usize>(
    key: &[u8; 32],
    aad: &[u8],
    blob: &CtBlob<PT_LEN>,
) -> Result<[u8; PT_LEN], AeadError> {
    let cipher = XChaCha20Poly1305::new_from_slice(key).unwrap();
    let xnonce = XNonce::from_slice(&blob.nonce);

    let mut pt = blob.ct;
    let tag = GenericArray::from_slice(&blob.tag);

    cipher.decrypt_in_place_detached(xnonce, aad, &mut pt, tag)?;
    Ok(pt)
}
