# Lifting Optimized Binaries to Canonical Compiler IR via Structure-Aware Retrieval and Iterative Verification

📄 Accepted at ACL 2026

---

## Overview

Lifting stripped and highly optimized binaries to a canonical compiler intermediate representation (IR) is essential for program analysis and cross-platform migration when source code is unavailable. However, aggressive compiler optimizations significantly distort control-flow and data-flow structures, making both rule-based and LLM-based decompilation approaches brittle and unreliable.

We present **BRIDGE**, a system that reliably lifts optimized binaries into analysis-friendly LLVM IR. Our approach integrates **structure-aware retrieval** with **feedback-driven iterative verification** to address both structural reconstruction and semantic correctness.

Specifically, BRIDGE introduces:

- **Control-flow-aware Retrieval-Augmented Generation (RAG):**  
  We construct a structure-aligned knowledge base of assembly–IR pairs using pseudo-probe instrumentation, enabling retrieval at the control-flow granularity to guide initial IR generation.

- **Iterative Verification and Refinement:**  
  We design a feedback loop that leverages compiler diagnostics, static analysis, and runtime execution to iteratively repair structural and semantic errors, significantly improving re-executability.

We evaluate BRIDGE on **HumanEval-Decompile** and **MBPP**, lifting both **x86-64** and **ARM64** binaries to LLVM IR.  
Our method consistently outperforms seven baselines, achieving **over 30% higher re-executability** compared to the strongest general-purpose LLM baseline.

---


### **Evaluation_Datasets/**

This directory contains **all evaluation datasets** used in the experiments (as described in **Section 6** of the paper). As well as the corresponding decompilation results of the evaluation sets.
