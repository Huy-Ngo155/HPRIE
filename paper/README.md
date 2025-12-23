# HPRIE Technical Paper

This directory contains the formal research paper for **HPRIE: A NUMA-Aware Lock-Free Inference Engine for Low-Latency AI Serving**.

## 📄 Read the Paper
Click the link below to view the full PDF:
- [**Download Technical Paper (PDF)**](./HPRIE__A_NUMA_Aware_Lock_Free_Inference_Engine_for_Low_Latency_AI_Serving.pdf)

## 📌 Abstract
AI inference serving in production environments is constrained by systems-level bottlenecks such as lock contention and non-uniform memory access (NUMA) effects. [cite_start]This paper presents **HPRIE**, a high-performance inference engine integrating lock-free concurrency, explicit memory ordering, and NUMA-aware scheduling. [cite: 4, 6]

## 🛠 Key Contributions
- [cite_start]**NUMA-Aware Architecture**: Optimized for multi-core server environments. [cite: 17]
- [cite_start]**Lock-Free MPMC Pipeline**: Uses C++ atomics with formal progress guarantees (Linearizability & Lock-Freedom). [cite: 18, 46, 89, 90]
- [cite_start]**Predictable Latency**: Significantly reduces tail latency (P99) under high concurrency compared to mutex-based baselines. [cite: 67, 80]

---
*Authored by Huy Ngo (Independent Researcher)*
