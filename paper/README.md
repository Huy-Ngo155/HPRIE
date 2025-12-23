# HPRIE Technical Paper

[cite_start]This directory contains the formal research documentation for **HPRIE: A NUMA-Aware Lock-Free Inference Engine for Low-Latency AI Serving**[cite: 1, 122].

## 📄 Read the Paper
Click the link below to view or download the full technical paper (Standard 2-column academic format):
- [**Download Technical Paper (PDF)**](./HPRIE_Technical_Paper.pdf)

## 📌 Abstract
[cite_start]AI inference serving in production is often constrained by systems-level bottlenecks such as lock contention and non-uniform memory access (NUMA) effects[cite: 4, 127]. [cite_start]HPRIE integrates classical systems programming techniques—lock-free concurrency, explicit memory ordering, and NUMA-aware scheduling—into the inference pipeline[cite: 6, 128]. [cite_start]It is designed to provide predictable low-latency behavior under high contention[cite: 7, 129].

## 🛠 Key Technical Contributions
- [cite_start]**NUMA-Aware Architecture**: Optimized for multi-core server environments by explicitly binding worker threads and memory pools to specific NUMA nodes[cite: 17, 41, 142].
- [cite_start]**Lock-Free MPMC Pipeline**: Implements a Multi-Producer Multi-Consumer task queue using C++ atomics to eliminate kernel transitions and priority inversion[cite: 18, 46, 145].
- [cite_start]**Formal Progress Guarantees**: The core components satisfy **Linearizability** and **Lock-Freedom**, ensuring system-wide progress regardless of individual thread delays[cite: 87, 89, 90, 148].
- [cite_start]**Predictable P99 Latency**: Specifically engineered to stabilize tail latency by removing coarse-grained locks and reducing cross-NUMA memory traffic[cite: 67, 81, 146].

---
[cite_start]*Authored by Huy Ngo (Independent Researcher)* [cite: 2, 125]
