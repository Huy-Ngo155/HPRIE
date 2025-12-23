# HPRIE Technical Paper

This directory contains the formal research documentation for **HPRIE: A NUMA-Aware Lock-Free Inference Engine for Low-Latency AI Serving**.

## 📄 Read the Paper
Click the link below to view the technical paper:
- [**Download Technical Paper (PDF)**](./HPRIE_Technical_Paper.pdf)

## 📌 Abstract
AI inference serving in production environments is often constrained by systems-level bottlenecks such as lock contention and non-uniform memory access (NUMA) effects. HPRIE integrates lock-free concurrency, explicit memory ordering, and NUMA-aware scheduling to provide predictable low-latency behavior under contention.

## 🛠 Key Technical Contributions
- **NUMA-Aware Architecture**: Optimized for multi-core environments by preserving memory locality on multi-socket systems.
- **Lock-Free MPMC Pipeline**: Utilizes Multi-Producer Multi-Consumer queues implemented with C++ atomic primitives to eliminate kernel transitions.
- **Formal Progress Guarantees**: The architecture satisfies **Linearizability** and **Lock-Freedom**, ensuring continuous system-wide progress.
- **Predictable Latency**: Designed to treat inference serving as a systems-level resource management problem to maintain stability.

---
*Authored by Huy Ngo (Independent Researcher)*
