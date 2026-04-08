# MXFP4 GEMM optimization on AMD MI35XX

**MXFP4 GEMM** is an optimization technique that reduces memory bandwidth in matrix multiplication by combining **on-the-fly FP4 quantization of activations (A)** with **pre-quantized FP4 weights (B)**.

---

## **Overview**
This project implements a fused pipeline:

- Quantizes **bf16 activations (A)** → MXFP4 (FP4 + scales)  
- Uses **pre-quantized & pre-shuffled weights (B)**  
- Performs **FP4 × FP4 GEMM → bf16 output**

Designed for **memory-bound workloads** where bandwidth is the bottleneck.

---

## **Core Workflow**
**bf16 A + MXFP4 B → Quantize A → Shuffle Layout → FP4 GEMM → bf16 Output**

---

## **Benefits**

- **Bandwidth Reduction** → ~4× smaller than bf16  
- **Kernel Fusion** → Eliminates extra memory passes  
- **Efficient Compute** → Operates on compressed data  
- **Practical Accuracy** → Minimal loss vs bf16  

---

## **Kernel Architecture**

1. **Quantize (A)**
   - Uses `_mxfp4_quant_op`
   - Block size = **32**
   - Packs **2 FP4 values per byte**

2. **Shuffle (Scales Layout)**
   - Writes scales directly in **GEMM-ready format**
   - No separate reorder kernel

3. **Store (Compressed Representation)**
   - FP4 data (`fp4x2`)
   - Block scales (`fp8_e8m0`)

4. **GEMM Execution**
   - `aiter.gemm_a4w4`
   - **FP4 × FP4 → bf16**

---

## **Why Kernel Fusion?**

**Without fusion:**
```
bf16 → quant → write → read → reorder → write → GEMM
```

**With fusion:**
```
bf16 → quant + reorder → GEMM
```

-> Fewer memory passes  
-> Higher effective throughput  

---

## **Tools Used**

- **Triton** → Custom GPU kernel  
- **PyTorch (ROCm)** → Tensor ops  
- **aiter** → FP4 GEMM + dtypes  
- **ROCm stack** → AMD GPU execution  

---

## **Quantization Strategy**

- **Format**: MXFP4 (E2M1)  
- **Block size**: 32  
- **Scaling**: per-block (E8M0)  
- **Packing**: 2 FP4 values per byte  

Special handling:
- Padding uses **127 (neutral scale)**  
- Layout matches GEMM requirements  

---

## **Design Trade-offs**

- **FP4 vs FP8** → Higher compression, slightly lower precision  
- **Fusion vs Modularity** → Better performance, less flexibility  
- **Block scaling vs global scaling** → Better accuracy  
- **Heuristics vs autotuning** → Simpler implementation  

---

## **Notes**

- Optimized for **AMD MI3xx GPUs**  
- Best for **memory-bound GEMM workloads**  
- Focused on **performance experimentation**  

---

## **Future Improvements**

- Autotuning kernel configs  
- Further fusion (quant + GEMM)  
- FP4 end-to-end pipelines  
- Better scheduling strategies  

---

## **Time and Trade-offs**

- Focused on **performance over generality**  
- Reused **production quant primitive**  
- Avoided heavy frameworks  
- Prioritized **fusion + memory efficiency**  
