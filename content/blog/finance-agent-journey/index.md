+++
title = 'Building a Finance Agent and Dataset: From Research Note to Replication'
date = 2026-06-02
categories = ["LLM"]
[params]
subtitle = "Reconstructing Fin-R1-Data, and what that means for an auditable finance agent"
tech_meta = "Fin-R1-DATA / DEEPSEEK-R1-671B / QWEN2.5-72B"
math = false
+++

This post documents the end-to-end thinking behind reproducing a Fin-R1-style financial reasoning dataset, and what that implies for building a finance agent that can answer questions with genuine chain-of-thought reasoning instead of backward-engineered justifications.

## Why replicate Fin-R1-Data?

The official Fin-R1-Data dataset is not publicly available. The authors at SUFE-AIFLM-Lab promised a March 2025 release but never followed through. That leaves practitioners with three options:

1. Reconstruct it from the raw open sources using the described two-stage pipeline.
2. Use a close substitute such as `BUPT-Reasoning-Lab/FinanceReasoning`.
3. Contact the authors and reference the open GitHub issues.

We chose option (1). The goal is a 5,000-example CoT dataset with DeepSeek Pro API as the blind teacher and Qwen 2.6 Plus as the judge.

## What Fin-R1-Data contains

The paper describes 60,091 bilingual (Chinese–English) financial reasoning samples. They are split into four categories:

| Category | Share | Approx. Samples | Content |
| --- | --- | --- | --- |
| Financial basic business knowledge | ~50% | ~31K | Regulatory compliance, domain knowledge |
| Financial advanced business knowledge | ~25% | ~15K | Numerical reasoning, sentiment, causal extraction |
| Financial professional knowledge | ~19% | ~11K+ | Terminology, calculations, postgraduate exam Qs |
| Financial code | <1% | 152 | Quantitative trading scripts, risk model code |

The raw sources span Finance-500K, FinCorpus, ConvFinQA, FinQA, FinanceIQ, TFNS, FinCUGE, plus closed or partially-closed sources (FinPEE, Ant-Finance, FinanceQT).

## The replication pipeline

The core methodology is uniform across datasets:

- Extract questions only.
- Generate reasoning + answer via DeepSeek-R1-671B (temperature 0.6).
- Filter using Qwen2.5-72B-Instruct: first by answer correctness, then by reasoning quality on seven dimensions.
- Keep a sample only if it scores 7/7.

That strictness explains the dramatic retention differences:

| Dataset | Original | Processed | Retention |
| --- | --- | --- | --- |
| Finance-500K | 518,190 | 11,300 | 2.2% |
| FinCorpus | 235,210 | 29,288 | 12.5% |
| ConvFinQA | 14,120 | 7,629 | 54.0% |
| FinQA | 8,280 | 2,948 | 35.6% |

Pre-verified QA datasets like ConvFinQA and FinQA survive at much higher rates than raw corpora or large-scale synthetic collections. The variation is not in method; the same two-stage filter is applied to every source.

## Why the answer must stay hidden from the teacher

One subtle but critical detail: `reference_answer` should never be included in the prompt sent to DeepSeek. It must be added to the record later, after distillation, so that the judge can compare the model-generated answer against the ground truth.

> Including the answer in the prompt turns reasoning into backward engineering. You end up with a model that guesses first and then constructs plausible steps, instead of genuinely reasoning step by step.

The correct flow is:

1. Load raw row, extract `question` only.
2. Send question to DeepSeek R1 blind.
3. Collect `reasoning` and `model_answer`.
4. Add `reference_answer` from the original dataset metadata.
5. Run Qwen judge on answer correctness and reasoning quality.

## What this means for the finance agent

A dataset built this way trains the model to produce reasoning before reaching a conclusion. That matters for finance tasks where the reasoning must be auditable, the intermediate steps should be checkable, and the answer must still be numerically correct.

The practical constraints are also real: the full paper setup needs massive GPU resources for DeepSeek-R1-671B and Qwen2.5-72B-Instruct. For our local replication we will:

- Use DeepSeek Pro API instead of local inference.
- Use Qwen 2.6 Plus API as the judge.
- Target 5,000 examples rather than the full 60K.
- Focus on the open-source portions that can be reconstructed faithfully.

## Current status

As of 2026-06-02, the repo already contains converted prompt files for FinQA, ConvFinQA, TFNS, Finance-500K, and FinanceQT. The next step is to run the blind teacher and judge against those prompts, collect the passes, and assemble the final CoT dataset for agent training.
