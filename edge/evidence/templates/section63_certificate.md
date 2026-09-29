> **TEMPLATE — NOT FOR USE WITHOUT LEGAL REVIEW.**
> The legal wording in this certificate is a working template prepared for a technical
> demonstration. It must be reviewed, and where necessary rewritten, by a qualified legal
> practitioner against the current text of the Bharatiya Sakshya Adhiniyam, 2023 (Section 63 and
> its Schedule) and any applicable rules before it is signed, filed or relied on in any
> proceeding. The technical annexures below are generated from the evidence store and are
> accurate as of the generation time shown.
>
> Editing guide: everything in this file is plain Markdown. Words starting with `$` are filled in
> by `edge/evidence/certificate.py`; do not rename them. All other text may be changed freely.

# CERTIFICATE UNDER SECTION 63(4)(c) OF THE BHARATIYA SAKSHYA ADHINIYAM, 2023

**In respect of electronic records produced by the TRUEWATCH edge surveillance system**

Certificate reference: `$certificate_id`
Generated: $generated_at
Records covered: $record_scope

---

## PART A — Statement of the person in charge of the device

I, **$partA_name**, $partA_designation, state as follows:

1. I am the person in lawful charge of the computer device described in Section 1 below, or of the
   management of the activities for which it is used, during the period in which the electronic
   records described in Section 2 were produced.

2. The electronic records described in Section 2 (the "records") were produced by that device in
   the ordinary course of its use for border surveillance at the location stated, during a period
   over which the device was regularly used to record, store and process information for that
   purpose.

3. Throughout the material part of that period the device was operating properly. Every period in
   which it was not operating, or in which recording was not sealed into the chain, is disclosed in
   Section 5 of this certificate as a logged gap, and did not affect the accuracy of the records.

4. The information contained in the records was fed into the device in the ordinary course of
   those activities, by camera capture, without manual intervention.

5. The records are identified by the hash values set out in Section 3, computed with the SHA-256
   algorithm at the time of capture, and I certify that the copies produced correspond to those
   hash values.

6. The statements in this certificate are true to the best of my knowledge and belief.

Place: $place
Date: $date

Signature: ______________________________
Name: $partA_name
Designation: $partA_designation

---

## PART B — Statement of the expert

I, **$partB_name**, $partB_designation, having examined the device, the evidence store and the
records described below, state that:

1. I have independently recomputed the SHA-256 value of each record from the file produced, using
   the procedure in Section 4, and obtained the values set out in Section 3.

2. I have verified the chaining of each record to its predecessor, the inclusion of each record in
   the Merkle root for its day, and the Ed25519 signature on that root against the public key
   registered at sector headquarters, with the results set out in Sections 3 and 4.

3. On the basis of that examination the records are, in my opinion, the records produced by the
   device at the times stated, unaltered since they were sealed. $verification_opinion

Place: $place
Date: $date

Signature: ______________________________
Name: $partB_name
Designation: $partB_designation

---

## Section 1 — The device

| Field | Value |
|---|---|
| Device identifier | `$device_id` |
| Description | $device_description |
| Signing key | Ed25519, key id `$key_id` |
| Public key (raw, hex) | `$public_key_hex` |
| Public key registered at sector HQ | $hq_key_status |
| Day boundary | IST calendar day (UTC$day_tz) |

## Section 2 — The records

$record_table

## Section 3 — Hash values and chain links

$hash_table

## Section 4 — The process that produced and sealed the records

1. **Hashing at capture.** Each clip is hashed with SHA-256 while it is being written: every byte
   passes through the hash in the same operation that writes it to storage. The digest input is the
   clip's bytes followed by the tag `TWCLIP1`, the frame count and the byte length (each as an
   unsigned 64-bit big-endian integer).
2. **Chaining.** Each record is entered as a numbered block whose hash is SHA-256 over a fixed,
   documented serialisation of the block number, clip identifier, camera identifier, start, end
   and sealing times, the clip hash and the previous block's hash. Altering, removing or
   reordering any record changes every later block hash.
3. **Daily root.** At the end of each day the day's block hashes are combined into a single
   Merkle root (SHA-256, leaf and node hashes domain-separated, an odd node promoted unchanged).
4. **Signature and mirror.** The root, the block range it covers and the gap log as it then stood
   are signed on the device with an Ed25519 key generated on the device, and the signed root is
   copied to the sector HQ mirror, where the device's public key was registered.
5. **Verification.** `python -m evidence.verify` repeats every computation above from the files
   on disk and reports any mismatch. Its result at the time this certificate was generated:

$verification_summary

$proof_section

## Section 5 — Logged gaps in the chain

$gap_table

---

*Generated by the TRUEWATCH evidence module (`edge/evidence/certificate.py`). The template text is
maintained in `edge/evidence/templates/section63_certificate.md`.*
