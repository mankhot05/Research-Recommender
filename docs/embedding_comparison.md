# Embedding comparison

Each cell is `title (score)`. Scores are cosine similarity within one model; don't compare them across columns, only the rankings.

## 1. how do I evaluate the retrieval quality of a RAG system?

| Rank | openai | scincl | specter |
|---|---|---|---|
| 1 | RAGGED: Towards Informed Design of Scalable and Stable RAG Systems (0.674) | RAGGED: Towards Informed Design of Scalable and Stable RAG Systems (0.879) | REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources (0.779) |
| 2 | Ragas: Automated Evaluation of Retrieval Augmented Generation (0.662) | REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources (0.864) | Ragas: Automated Evaluation of Retrieval Augmented Generation (0.773) |
| 3 | RAG vs. GraphRAG: A Systematic Evaluation and Key Insights (0.648) | Observations on Building RAG Systems for Technical Documents (0.862) | Streamlining Conformal Information Retrieval via Score Refinement (0.769) |
| 4 | AccurateRAG: A Framework for Building Accurate Retrieval-Augmented Question-Answering Applications (0.638) | Evaluating Multi-Hop Reasoning in RAG Systems: A Comparison of LLM-Based Retriever Evaluation Strategies (0.859) | Evaluating Multi-Hop Reasoning in RAG Systems: A Comparison of LLM-Based Retriever Evaluation Strategies (0.768) |
| 5 | Comprehensive and Practical Evaluation of Retrieval-Augmented Generation Systems for Medical Question Answering (0.633) | The Effect of Text Chunk Size on Retrieval-Augmented Generation Performance (0.859) | RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving (0.760) |

## 2. reference-free evaluation metrics for retrieval-augmented generation

| Rank | openai | scincl | specter |
|---|---|---|---|
| 1 | Ragas: Automated Evaluation of Retrieval Augmented Generation (0.620) | Ragas: Automated Evaluation of Retrieval Augmented Generation (0.934) | A Survey of Multimodal Retrieval-Augmented Generation (0.880) |
| 2 | Meta-prompting Optimized Retrieval-augmented Generation (0.616) | A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation (0.924) | Self-adaptive Multimodal Retrieval-Augmented Generation (0.873) |
| 3 | Active Retrieval Augmented Generation (0.599) | E2E-AFG: An End-to-End Model with Adaptive Filtering for Retrieval-Augmented Generation (0.922) | Ragas: Automated Evaluation of Retrieval Augmented Generation (0.871) |
| 4 | Corrective Retrieval Augmented Generation (0.598) | Corrective Retrieval Augmented Generation (0.918) | CogPlanner: Unveiling the Potential of Agentic Multimodal Retrieval Augmented Generation with Planning (0.837) |
| 5 | A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation (0.596) | DF-RAG: Query-Aware Diversity for Retrieval-Augmented Generation (0.916) | Overview of the TREC 2025 Retrieval Augmented Generation (RAG) Track (0.831) |

## 3. reducing hallucination in retrieval-augmented question answering

| Rank | openai | scincl | specter |
|---|---|---|---|
| 1 | Knowledge Graph-Guided Retrieval Augmented Generation (0.590) | Context Quality Matters in Training Fusion-in-Decoder for Extractive Open-Domain Question Answering (0.908) | AccurateRAG: A Framework for Building Accurate Retrieval-Augmented Question-Answering Applications (0.859) |
| 2 | Taming a Retrieval Framework to Read Images in Humanlike Manner for Augmenting Generation of MLLMs (0.587) | Structure Guided Retrieval-Augmented Generation for Factual Queries (0.907) | A Reproducibility Study of Metacognitive Retrieval-Augmented Generation (0.851) |
| 3 | MultiRAG: A Knowledge-guided Framework for Mitigating Hallucination in Multi-source Retrieval Augmented Generation (0.584) | Prompt-Based Abstention Fails Under Misleading Context: A Controlled Study of Small Frozen RAG Models (0.903) | Context Awareness Gate For Retrieval Augmented Generation (0.847) |
| 4 | Structure Guided Retrieval-Augmented Generation for Factual Queries (0.570) | Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style (0.898) | From Questions to Insightful Answers: Building an Informed Chatbot for University Resources (0.832) |
| 5 | Optimizing Query Generation for Enhanced Document Retrieval in RAG (0.545) | AccurateRAG: A Framework for Building Accurate Retrieval-Augmented Question-Answering Applications (0.897) | Context Quality Matters in Training Fusion-in-Decoder for Extractive Open-Domain Question Answering (0.821) |

## 4. sequential recommendation with transformers

| Rank | openai | scincl | specter |
|---|---|---|---|
| 1 | DACSR: Decoupled-Aggregated End-to-End Calibrated Sequential Recommendation (0.441) | PowerMat: context-aware recommender system without user item rating values that solves the cold-start problem (0.898) | AgenticRec: A Recommendation-Oriented Agentic Framework with Progressive Tool-Integrated Reasoning Optimization (0.866) |
| 2 | A Comprehensive Evaluation of Transformer-Based Question Answering Models and RAG-Enhanced Design (0.435) | Quantitative analysis of Matthew effect and sparsity problem of recommender systems (0.896) | A Survey on Reinforcement Learning for Recommender Systems (0.861) |
| 3 | UniSAR: Modeling User Transition Behaviors between Search and Recommendation (0.426) | Managing Popularity Bias in Recommender Systems with Personalized Re-ranking (0.889) | SSLRec: A Self-Supervised Learning Framework for Recommendation (0.852) |
| 4 | Pre-trained Language Model and Knowledge Distillation for Lightweight Sequential Recommendation (0.423) | Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation (0.884) | RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms (0.849) |
| 5 | Time is of the Essence: a Joint Hierarchical RNN and Point Process Model for Time and Item Predictions (0.423) | Overhead-free User-side Recommender Systems (0.883) | Multi-Interest Recommendation: A Survey (0.849) |

## 5. cold-start problem in recommender systems

| Rank | openai | scincl | specter |
|---|---|---|---|
| 1 | Offline Meta-level Model-based Reinforcement Learning Approach for Cold-Start Recommendation (0.596) | PowerMat: context-aware recommender system without user item rating values that solves the cold-start problem (0.936) | Review of Clustering-Based Recommender Systems (0.910) |
| 2 | PowerMat: context-aware recommender system without user item rating values that solves the cold-start problem (0.585) | Review of Clustering-Based Recommender Systems (0.917) | PowerMat: context-aware recommender system without user item rating values that solves the cold-start problem (0.900) |
| 3 | Ultra Fast Warm Start Solution for Graph Recommendations (0.537) | Knowledge Transfer via Pre-training for Recommendation: A Review and Prospect (0.917) | Ultra Fast Warm Start Solution for Graph Recommendations (0.897) |
| 4 | Reinforcement Learning based Recommender System using Biclustering Technique (0.535) | Quantitative analysis of Matthew effect and sparsity problem of recommender systems (0.915) | Quantitative analysis of Matthew effect and sparsity problem of recommender systems (0.896) |
| 5 | Knowledge Transfer via Pre-training for Recommendation: A Review and Prospect (0.529) | A Connection-Centric Survey of Recommender Systems Research (0.906) | A Survey Paper on Recommender Systems (0.888) |
