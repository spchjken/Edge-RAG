# 📝 Edge-RAG Pipeline V7 Calibration Decision Log

- **Generated:** 2026-08-31 06:20:26

## Stage 1 (Pool)

| Config ID | Display Name | Strict@10 | DocRec@10 | Strict@50 | DocRec@50 | MRR@10 | Mean Latency | Starved Aspects |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pool_strat_salience__N_2500` | Pool Strategy: salience (N=2500) | 62.77% | **49.68%** | 74.66% | 60.97% | 0.4814 | 142.60ms | 0.00 |
| `pool_strat_idf__N_2500` | Pool Strategy: idf (N=2500) | 62.58% | **49.51%** | 74.63% | 61.07% | 0.4809 | 95.57ms | 0.00 |
| `pool_strat_random__N_2500` | Pool Strategy: random (N=2500) | 62.54% | **49.45%** | 74.86% | 61.09% | 0.4817 | 108.80ms | 0.00 |
| `pool_size_coverage__N_5000` | Pool Size: coverage (N=5000) | 59.65% | **47.89%** | 72.25% | 59.89% | 0.4493 | 135.59ms | 0.00 |
| `pool_size_coverage__N_2500` | Pool Size: coverage (N=2500) | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 68.86ms | 0.00 |
| `pool_strat_coverage__N_2500` | Pool Strategy: coverage (N=2500) | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 69.07ms | 0.00 |
| `pool_size_coverage__N_500` | Pool Size: coverage (N=500) | 59.25% | **47.73%** | 71.94% | 59.61% | 0.4487 | 30.95ms | 0.00 |
| `pool_size_coverage__N_1000` | Pool Size: coverage (N=1000) | 59.25% | **47.66%** | 72.02% | 59.70% | 0.4484 | 38.60ms | 0.00 |

---

## Stage 2 (Gate)

| Config ID | Display Name | Strict@10 | DocRec@10 | Strict@50 | DocRec@50 | MRR@10 | Mean Latency | Starved Aspects |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gate_dtau_+0.25__beta_1.0__single` | Gate d_tau=+0.25 (beta=1.0, single) | 60.25% | **48.20%** | 72.68% | 60.27% | 0.4529 | 31.43ms | 6.68 |
| `gate_dtau_+0.20__beta_1.0__single` | Gate d_tau=+0.20 (beta=1.0, single) | 60.10% | **48.12%** | 72.49% | 60.19% | 0.4495 | 33.75ms | 3.81 |
| `gate_dtau_+0.15__beta_1.0__single` | Gate d_tau=+0.15 (beta=1.0, single) | 59.77% | **47.94%** | 72.32% | 60.10% | 0.4487 | 37.07ms | 1.10 |
| `gate_dtau_+0.10__beta_1.0__single` | Gate d_tau=+0.10 (beta=1.0, single) | 59.61% | **47.92%** | 72.11% | 59.88% | 0.4488 | 42.72ms | 0.04 |
| `gate_dtau_+0.05__beta_1.0__single` | Gate d_tau=+0.05 (beta=1.0, single) | 59.55% | **47.89%** | 72.26% | 59.88% | 0.4491 | 52.19ms | 0.00 |
| `gate_beta_0.65__dtau_0.0__single` | Gate beta=0.65 (d_tau=0.0, single) | 59.50% | **47.88%** | 72.08% | 59.80% | 0.4495 | 51.24ms | 0.00 |
| `gate_beta_0.00__dtau_0.0__single` | Gate beta=0.00 (d_tau=0.0, single) | 59.52% | **47.85%** | 72.36% | 60.13% | 0.4452 | 42.41ms | 0.37 |
| `gate_dtau_-0.15__beta_1.0__single` | Gate d_tau=-0.15 (beta=1.0, single) | 59.53% | **47.84%** | 72.16% | 59.73% | 0.4482 | 161.75ms | 0.00 |
| `gate_dtau_-0.10__beta_1.0__single` | Gate d_tau=-0.10 (beta=1.0, single) | 59.53% | **47.84%** | 72.16% | 59.74% | 0.4480 | 138.31ms | 0.00 |
| `gate_variant_soft_reweight__dtau_0.0__beta_1.0` | Gate Variant: soft_reweight | 59.48% | **47.83%** | 72.14% | 59.79% | 0.4485 | 86.61ms | 0.00 |
| `gate_beta_0.50__dtau_0.0__single` | Gate beta=0.50 (d_tau=0.0, single) | 59.46% | **47.82%** | 72.14% | 59.84% | 0.4488 | 45.69ms | 0.13 |
| `gate_dtau_+0.00__beta_1.0__single` | Gate d_tau=+0.00 (beta=1.0, single) | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 72.15ms | 0.00 |
| `gate_beta_1.00__dtau_0.0__single` | Gate beta=1.00 (d_tau=0.0, single) | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 71.64ms | 0.00 |
| `gate_variant_single__dtau_0.0__beta_1.0` | Gate Variant: single | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 72.24ms | 0.00 |
| `gate_variant_two_gate__dtau_0.0__beta_1.0` | Gate Variant: two_gate | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4484 | 76.02ms | 0.00 |
| `gate_dtau_-0.05__beta_1.0__single` | Gate d_tau=-0.05 (beta=1.0, single) | 59.41% | **47.79%** | 72.16% | 59.76% | 0.4477 | 104.72ms | 0.00 |
| `gate_dtau_-0.25__beta_1.0__single` | Gate d_tau=-0.25 (beta=1.0, single) | 59.45% | **47.76%** | 72.16% | 59.73% | 0.4479 | 184.24ms | 0.00 |
| `gate_dtau_-0.20__beta_1.0__single` | Gate d_tau=-0.20 (beta=1.0, single) | 59.45% | **47.75%** | 72.16% | 59.73% | 0.4477 | 176.30ms | 0.00 |

---

## Stage 3 (Budget & Sparsity)

| Config ID | Display Name | Strict@10 | DocRec@10 | Strict@50 | DocRec@50 | MRR@10 | Mean Latency | Starved Aspects |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `alloc_softmax_1.0` | Allocation: softmax_1.0 | 59.48% | **47.83%** | 72.14% | 59.79% | 0.4484 | 72.86ms | 0.00 |
| `alloc_softmax_0.1` | Allocation: softmax_0.1 | 59.45% | **47.83%** | 72.24% | 59.84% | 0.4481 | 72.79ms | 0.00 |
| `budget_eta_-0.5__mu_0.50` | Budget eta=-0.5 (mu=0.50) | 59.48% | **47.83%** | 72.14% | 59.80% | 0.4480 | 71.38ms | 0.00 |
| `budget_mu_0.75__eta_0.0` | Budget mu_ceil=0.75 | 59.48% | **47.83%** | 72.14% | 59.80% | 0.4481 | 71.99ms | 0.00 |
| `budget_mu_1.00__eta_0.0` | Budget mu_ceil=1.00 | 59.48% | **47.82%** | 72.16% | 59.81% | 0.4482 | 71.72ms | 0.00 |
| `budget_mu_0.25__eta_0.0` | Budget mu_ceil=0.25 | 59.48% | **47.81%** | 72.20% | 59.80% | 0.4485 | 71.59ms | 0.00 |
| `budget_eta_+0.5__mu_0.50` | Budget eta=+0.5 (mu=0.50) | 59.48% | **47.81%** | 72.16% | 59.80% | 0.4485 | 71.06ms | 0.00 |
| `alloc_uniform` | Allocation: uniform | 59.45% | **47.80%** | 72.14% | 59.79% | 0.4482 | 71.15ms | 0.00 |
| `alloc_normalized_cosine` | Allocation: normalized_cosine | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 71.60ms | 0.00 |
| `budget_mu_0.50__eta_0.0` | Budget mu_ceil=0.50 | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 71.60ms | 0.00 |
| `budget_eta_+0.0__mu_0.50` | Budget eta=+0.0 (mu=0.50) | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 71.35ms | 0.00 |
| `sparsity_eps_0.000` | Mass Floor eps=0.000 | 59.41% | **47.80%** | 72.14% | 59.79% | 0.4481 | 71.56ms | 0.00 |
| `sparsity_eps_0.005` | Mass Floor eps=0.005 | 59.36% | **47.78%** | 71.97% | 59.77% | 0.4476 | 41.07ms | 0.00 |
| `sparsity_eps_0.020` | Mass Floor eps=0.020 | 59.36% | **47.78%** | 71.97% | 59.77% | 0.4477 | 41.16ms | 0.00 |
| `sparsity_eps_0.010` | Mass Floor eps=0.010 | 59.36% | **47.78%** | 71.97% | 59.77% | 0.4477 | 41.12ms | 0.00 |
| `sparsity_eps_0.050` | Mass Floor eps=0.050 | 59.36% | **47.78%** | 71.97% | 59.77% | 0.4477 | 41.04ms | 0.00 |
| `sparsity_eps_0.001` | Mass Floor eps=0.001 | 59.35% | **47.77%** | 72.09% | 59.79% | 0.4480 | 46.80ms | 0.00 |

---

## Stage 4 (Joint Freeze)

| Config ID | Display Name | Strict@10 | DocRec@10 | Strict@50 | DocRec@50 | MRR@10 | Mean Latency | Starved Aspects |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `v7_frozen_candidate_B` | V7 Candidate B (dtau=+0.10, mu=0.50, eps=0.010) | 59.60% | **47.90%** | 72.21% | 59.91% | 0.4491 | 31.22ms | 0.04 |
| `v7_frozen_candidate_A` | V7 Candidate A (dtau=+0.05, mu=0.50, eps=0.005) | 59.57% | **47.89%** | 72.12% | 59.87% | 0.4487 | 35.06ms | 0.00 |
| `v7_frozen_candidate_C` | V7 Candidate C (dtau=0.00, mu=0.50, eps=0.005) | 59.36% | **47.78%** | 71.97% | 59.77% | 0.4476 | 41.14ms | 0.00 |

---

