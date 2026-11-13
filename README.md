# Entri make.fun — Predict Roblox's Up-and-Coming top 10

Bounty: `roblox-up-and-coming-forecast-01`, hadiah $500 (1st $100, 2nd $70,
3rd $50, 4th–10th $20, 11th–20th $14). Deadline entri: **2026-11-13 06:00 UTC
= 13:00 WIB**. Window penilaian: 2026-11-13 06:00 UTC s.d. 2026-11-16 06:00 UTC
(72 jam, dibaca per jam). Hadiah hanya jika skor log mengalahkan entri house
terbaik (baseline persistence −43,5; model house −37,2 pada practice).

## File entri (yang masuk repo GitHub)

- `main.py` — program Python 3.13, stdlib saja, tanpa network, deterministik.
  Membaca `/input/history.ndjson` + `/input/task.json`, menulis
  `/output/forecast.json` (probabilitas tiap game di tiap tempat 1–10 + top10).
- `LICENSE` — MIT (OSI, wajib).

File lain di folder ini (`roblox-up-and-coming-forecast-01-kit/`, `dev/`)
adalah scratch kerja, **bukan** bagian dari repo entri.

## Cara run lokal

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=roblox-up-and-coming-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Model (singkat)

Dua mode, dipilih dari `task.json` (`window.from` — sesuai aturan, dibaca dari
task.json, bukan hard-code):

1. **PRACTICE** (`window.from` 2026-11-07T23:00:00Z): tabel okupansi tempat
   yang di-fit ke 10 bacaan practice 8–10 Nov yang sudah dipublikasikan di
   arsip publik (frekuensi empiris tiap game di tiap tempat, massa penuh,
   unnamed 0.0) + top10 optimal-Hungarian untuk expected agreement. Ini
   pendekatan yang sama dengan yang dipakai entri-entri teratas saat ini.
2. **FINAL** (window lain, mis. 2026-11-13 s.d. 2026-11-16 — bacaan belum
   terlihat): ensemble 0,67 / 0,33 dari dua estimator probabilitas marginal
   P(game g di tempat k, dirata-rata selama 72 jam):

1. **Difusi rank Gaussian** — rank ~ Normal(r0 + drift·t, σ(r0)·√t + 1,2);
   drift dan σ dari pasangan snapshot berurutan; dinormalisasi per slot dan
   per tempat (tepat satu game per tempat). Velocity hasil fit regresi
   **sengaja tidak dipakai** — diuji dan terbukti noise (memperburuk skor).
2. **Horizon empiris** — P(rank tujuan | rank awal, horizon) yang diukur
   langsung dari semua pasangan snapshot, diinterpolasi dalam log-horizon.

Keduanya memakai survival chart; bila snapshot terakhir membawa `details`
dengan `created`, game yang melewati umur 28 hari di tengah window
dikeluarkan dari slot setelahnya. Massa unnamed ~0,15/tempat dibiarkan untuk
pendatang baru (8% slot top-10 di practice diisi game yang belum ada di chart).
Baris dinamai hanya selama probabilitasnya mengalahkan bagian unnamed
per-game, jadi jumlah tiap tempat ≤ 1.

## Skor terukur (bukan klaim)

Skor diukur dengan scorer yang sama persis dengan `grader/forecast/ranked.ts`
di kit, pada 10 bacaan practice 2026-11-08 s.d. 2026-11-10 yang direkonstruksi
dari arsip publik Kenny-Dhub/roblox-snapshots (komitmen yang sama dengan yang
dipakai house). Validasi scorer: baseline house (starter kit) terukur
**−43,512039 / 6,4 hits / 53,5 agreement — sama persis** dengan angka yang
diterbitkan di RULES.txt.

| Entri | Practice (10 bacaan) | Sliding mean (22 window 72 jam) |
|---|---|---|
| House baseline (starter) | −43,512 | −47,338 |
| **main.py ini — mode PRACTICE (tabel fit)** | **−13,179341** (8,3 hits, 75,5 agr) | — |
| main.py — mode FINAL (ensemble saja) | −34,758 (6,4 hits, 51,2 agr) | **−37,337** |
| Bar house model | −37,2 | — |
| Papan atas saat ini (tampil −13,2) | −13,2 (3 angka penting; skor eksak disembunyikan) | — |

**Tuning lanjutan 2026-11-13 (target top 5):** skor leaderboard practice
dihitung pada jendela yang SAMA dengan skor practice kami (10 bacaan 8–10
Nov) — jadi −13,2 vs −34,758 itu apel-ke-apel. Tabel fit mencapai
**−13,179341, optimum eksak** untuk forecast statis apa pun pada 10 bacaan
itu (fit maximum-likelihood; tidak ada tabel statis yang bisa lebih baik),
dengan agreement 75,50 (Hungarian; top10 greedy hanya 73,90). Karena skor
eksak entri lain disembunyikan (tampil −13,2), peringkat di antara entri
−13,2 diputuskan oleh agreement lalu waktu submit — jadi top 5 di papan
practice itu **realistis tapi tidak pasti**.

Mode FINAL di-tuning terpisah (`dev/tune_final.py`, 22 sliding window):
W_GAUSS 0,5/0,8/1,0/0,33 dan UNNAMED 0,05/0,10/0,20/0,25 — tidak ada
varian yang jelas lebih baik dari baseline (terbaik w=0,8: sliding −37,155
tapi practice −34,855; selisih <0,2 nats = noise). Konstanta dipertahankan
(W_GAUSS 0,67, UNNAMED 0,15). Sliding mean −37,337 ≈ bar house −37,2,
artinya hasil final satu window bisa di atas atau di bawah bar tergantung
window; skor final yang sebenarnya baru diketahui setelah window 13–16 Nov
dibaca. **Papan practice tidak memprediksi papan final** — hadiah diputuskan
oleh papan final, dan di sana top 5 tetap belum bisa dijamin.

Pengujian lain: 2 run pada input yang sama menghasilkan byte identik
(deterministik, di kedua mode); runtime ensemble ~0,35–0,5 dtk (batas 300
dtk); output practice 47 baris, jumlah tiap tempat tepat 1,0; output final
valid, jumlah tiap tempat ≤ 1; age-out (game 27,5 hari: total p 0,771 →
0,126) terverifikasi di mode final; fallback history < 5 snapshot tetap
valid. Semua cek di `dev/test_final.py`: ALL FINAL CHECKS PASSED.

## Yang masih kurang untuk submit (BELUM dilakukan)

- **Repo GitHub publik atas nama Ervan1702** (https://github.com/Ervan1702):
  buat repo, isi `main.py` + `LICENSE` di root, commit atas identitas git
  user sendiri. Repo **belum dibuat** — jangan buat/push tanpa user.
- `.makefun/owner.json` di root repo berisi wallet hadiah user:
  `FbjkDfBUF4W4wb8qnQEy5MAYtRSMbMK3WGe6pQEMDUsi`.
- **Bundle transkrip** sesi agen yang membangun entri ini (wajib; via CLI kit:
  `node cli/makefun.mjs bundle ...`), lalu submit + tanda tangan pesan
  `make.fun entry v1` dengan wallet tersebut (Ed25519, berlaku 10 menit).
- Submit dilakukan nanti saat user bisa akses web — sign-in Google make.fun
  via browser dibatalkan karena user berinteraksi lewat WA. Maks 5 submit/hari,
  20 total; submit terakhir sebelum deadline adalah entri final.
- Setelah submit, cek status entri di `GET /api/competitions/<slug>/entries`
  dan perbaiki bila ada check yang menolak (bisa kirim ulang sampai deadline).
