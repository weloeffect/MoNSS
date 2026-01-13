# SLM2 Benchmark Evaluation Results

## Overview

This document presents the comprehensive evaluation results for **SLM2** (Knowledge Graph Verbalizer) tested on the `slm2_test.jsonl` dataset.

### Evaluation Date
**January 13, 2026**

### Dataset Statistics
- **Total Examples**: 3,528
- **1-Hop Examples**: 1,561 (44.3%)
- **2-Hop Examples**: 1,967 (55.7%)
- **True Facts (label=true)**: 1,851 (52.5%)
- **False Facts (label=false)**: 1,677 (47.5%)

---

## 📊 Key Metrics

### Overall Performance

| Metric | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| **Overall Accuracy** | 2,739 | 3,528 | **77.64%** |
| **Single-Hop Accuracy (1-hop)** | 1,202 | 1,561 | **77.00%** |
| **2-Hop Accuracy** | 1,537 | 1,967 | **78.14%** |

### Label-Specific Performance

| Metric | Correct | Total | Accuracy | Interpretation |
|--------|---------|-------|----------|----------------|
| **True-Fact Accuracy** | 1,403 | 1,851 | **75.80%** | Measures verbalization ability |
| **False-Fact Accuracy** | 1,336 | 1,677 | **79.67%** | Measures hallucination resistance |

---

## 🔍 Key Findings

### 1. **Strong Hallucination Resistance** ✅
- SLM2 achieves **79.67% accuracy** on false facts (unknown knowledge)
- This indicates good ability to say "I don't know" when appropriate
- Only 341 false confidence errors out of 1,677 false examples (20.3%)

### 2. **Competitive Multi-Hop Reasoning** ✅
- 2-hop accuracy (**78.14%**) is actually **higher** than 1-hop accuracy (77.00%)
- This is surprising and suggests the model handles compositional reasoning well
- Only 64 multi-hop-specific failures (3.3% of 2-hop errors)

### 3. **Room for Improvement in Verbalization** ⚠️
- True-fact accuracy is **75.80%**, slightly lower than false-fact accuracy
- 448 incorrect verbalizations out of 1,851 true facts (24.2%)
- Main issues: false confidence (saying "I don't know" for true facts), under-verbalization

### 4. **Error Distribution**
Total Errors: **789** (22.36% of all examples)

| Error Type | Count | Percentage of Errors |
|------------|-------|---------------------|
| **False Confidence** | 341 | 43.2% |
| **Other** | 303 | 38.4% |
| **Multi-Hop Failure** | 64 | 8.1% |
| **Wrong Decision** | 47 | 6.0% |
| **Under-Verbalization** | 34 | 4.3% |

---

## 📈 Detailed Analysis

### Error Categories Explained

1. **False Confidence (43.2% of errors)**
   - Model verbalizes something when it should say "I don't know"
   - Example: For `label=false`, model outputs a sentence instead of "I don't know."
   - **Impact**: Reduces false-fact accuracy

2. **Other (38.4% of errors)**
   - Semantic mismatches or incorrect verbalization of true facts
   - Does not match gold output despite being a legitimate verbalization attempt
   - **Impact**: Reduces true-fact accuracy

3. **Multi-Hop Failure (8.1% of errors)**
   - Specifically fails on 2-hop reasoning where 1-hop would succeed
   - Says "I don't know" for valid 2-hop facts
   - **Impact**: Reduces 2-hop accuracy

4. **Wrong Decision (6.0% of errors)**
   - Says "I don't know" for valid 1-hop facts
   - **Impact**: Reduces 1-hop and true-fact accuracy

5. **Under-Verbalization (4.3% of errors)**
   - Uses placeholders like "probably" or leaves out the final value
   - Example: "Agra Airport probably has location that has is part of." (missing "Bundelkhand")
   - **Impact**: Reduces true-fact accuracy

---

## 🎯 Evaluation Contract

The evaluation followed strict criteria:

### For `label=false` (Unknown Facts):
- ✅ **Correct**: Exactly "I don't know."
- ❌ **Incorrect**: Any other output (verbalization, placeholders, variations)

### For `label=true` (Known Facts):
- ✅ **Correct**: Faithful verbalization matching gold output
- ❌ **Incorrect**: 
  - Says "I don't know"
  - Contains placeholders (SOME_VALUE, INTERMEDIATE, "probably")
  - Does not match gold output

---

## 💡 Recommendations

### Strengths to Maintain:
1. ✅ Good hallucination resistance (79.67%)
2. ✅ Strong 2-hop reasoning (78.14%)
3. ✅ Balanced performance across hop counts

### Areas for Improvement:
1. ⚠️ **Reduce False Confidence**: 341 cases where model verbalized false facts
   - Consider stricter confidence thresholds
   - Better uncertainty detection

2. ⚠️ **Improve True-Fact Verbalization**: 448 errors on true facts
   - Better handling of edge cases
   - More consistent verbalization patterns

3. ⚠️ **Eliminate Placeholders**: 34 under-verbalization errors
   - Remove "probably" from outputs
   - Ensure complete verbalization or strict "I don't know"

---

## 📁 Generated Files

The evaluation produced the following outputs:

1. **`slm2_evaluation_results.json`**
   - Complete results with all metrics
   - Detailed error examples by category
   - Machine-readable format for further analysis

2. **`slm2_error_report.txt`**
   - Human-readable error analysis
   - Sample error examples for each category
   - Useful for debugging and improvement

3. **`README_SLM2_EVALUATION.md`** (this file)
   - High-level summary and findings
   - Recommendations for improvement

---

## 🔧 Reproducibility

To reproduce these results:

```bash
cd benchmark
python slm2_evaluation.py
```

**Requirements:**
- Python 3.7+
- Input: `slm2_output.jsonl` (predictions)
- Reference: `../data/test/slm2_test.jsonl` (test set)

---

## 📊 Comparison Baseline

For context, a perfect model would achieve:
- **100% Overall Accuracy**
- **100% on both 1-hop and 2-hop**
- **100% on both true-facts and false-facts**

SLM2 achieves ~77-79% across all categories, showing consistent performance with slight edge on 2-hop reasoning.

---

## Conclusion

SLM2 demonstrates **solid performance** as a knowledge graph verbalizer with:
- ✅ Good baseline accuracy (~77-79%)
- ✅ Strong hallucination resistance
- ✅ Effective multi-hop reasoning
- ⚠️ Room for improvement in reducing false confidence and improving verbalization quality

The model shows balanced capabilities across different reasoning depths and maintains good performance on both verbalizing known facts and refusing to hallucinate unknown ones.

---

**Evaluation Script**: [`slm2_evaluation.py`](slm2_evaluation.py)  
**Test Dataset**: [`../data/test/slm2_test.jsonl`](../data/test/slm2_test.jsonl)  
**Model Outputs**: [`slm2_output.jsonl`](slm2_output.jsonl)
