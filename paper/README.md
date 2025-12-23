# HPRIE Technical Paper

This directory contains the formal research documentation for **HPRIE: A NUMA-Aware Lock-Free Inference Engine for Low-Latency AI Serving**.

## 📄 Read the Paper
Click the link below to view or download the full technical paper:
- [**Download Technical Paper (PDF)**](./HPRIE_Technical_Paper.pdf)

## 📌 Abstract
[cite_start]AI inference serving in production environments is constrained by systems-level bottlenecks such as lock contention and non-uniform memory access (NUMA) effects[cite: 6]. [cite_start]HPRIE integrates lock-free concurrency, explicit memory ordering, and NUMA-aware scheduling to provide predictable low-latency behavior under contention[cite: 7, 8].

## 🛠 Key Technical Contributions
- [cite_start]**NUMA-Aware Architecture**: Optimized for multi-core server environments by preserving memory locality on multi-socket systems[cite: 14, 17].
- [cite_start]**Lock-Free MPMC Pipeline**: Utilizes Multi-Producer Multi-Consumer queues implemented with C++ atomic primitives to eliminate kernel transitions and reduce tail latency (P99)[cite: 20, 24, 25].
- [cite_start]**Formal Progress Guarantees**: The system architecture satisfies **Linearizability** and **Lock-Freedom**, ensuring continuous progress without the "convoy effects" common in mutex-based systems[cite: 27, 28].
- [cite_start]**Predictable Latency**: Specifically designed to treat inference serving as a systems-level resource management problem to maintain stability under load spikes[cite: 11, 16].

---
*Authored by Huy Ngo (Independent Researcher)*
