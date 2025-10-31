#!/usr/bin/env python3
"""
Demo: Balanced Sequential vs Concurrent Multi-Model Performance

This demo uses balanced workloads to better showcase concurrent execution.
Uses longer texts for embeddings and multiple completions to ensure both
models are working simultaneously for a significant duration.
"""

import asyncio
import time
from typing import List, Dict
import httpx
from datetime import datetime


BASE_URL = "http://localhost:8200/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GENERATION_MODEL = "Qwen/Qwen3-4B-Thinking-2507-FP8"


# Longer texts for embedding to increase processing time (32 texts for batch processing)
EMBEDDING_TEXTS = [
    """Artificial intelligence represents one of the most transformative technologies of our time. 
    Machine learning algorithms enable computers to learn patterns from vast amounts of data, 
    leading to breakthroughs in areas like computer vision, natural language processing, and 
    robotics. Deep learning, powered by neural networks with multiple layers, has revolutionized 
    how we approach complex problems in image recognition, speech synthesis, and language translation.""",
    
    """The future of computing lies in the convergence of multiple technologies including quantum 
    computing, edge computing, and neuromorphic architectures. These advances promise to solve 
    problems that are currently intractable with classical computers, from drug discovery and 
    climate modeling to optimization problems in logistics and finance. The synergy between AI 
    and quantum computing could unlock entirely new paradigms of computational capability.""",
    
    """Natural language processing has evolved dramatically in recent years, moving from rule-based 
    systems to sophisticated neural architectures like transformers. Models trained on massive text 
    corpora can now understand context, generate coherent text, translate between languages, and 
    even engage in meaningful conversations. This technology is transforming how humans interact 
    with computers and access information in the digital age.""",
    
    """Computer vision systems can now recognize objects, detect anomalies, and understand scenes 
    with superhuman accuracy in many domains. From autonomous vehicles navigating city streets to 
    medical imaging systems detecting diseases, these visual AI systems are becoming integral to 
    modern infrastructure. The combination of high-resolution sensors, powerful GPUs, and advanced 
    neural architectures continues to push the boundaries of what machines can see and understand.""",
    
    """The ethical implications of artificial intelligence development require careful consideration 
    as these systems become more powerful and pervasive. Issues of bias, transparency, accountability, 
    and the societal impact of automation demand thoughtful governance frameworks. Ensuring that AI 
    systems are developed responsibly, with human values at their core, is one of the most important 
    challenges facing technologists, policymakers, and society as a whole.""",
    
    """Data science combines statistical analysis, machine learning, and domain expertise to extract 
    insights from complex datasets. The exponential growth in data generation from IoT devices, 
    social media, and digital transactions creates both opportunities and challenges. Advanced 
    analytics techniques enable organizations to make data-driven decisions, predict trends, and 
    optimize operations across industries from healthcare to finance to manufacturing.""",
    
    """Blockchain technology introduces decentralized trust mechanisms that could transform how we 
    handle digital transactions and record-keeping. Beyond cryptocurrencies, distributed ledger 
    technologies offer potential applications in supply chain management, digital identity verification, 
    and secure data sharing. Smart contracts enable programmable agreements that execute automatically 
    when conditions are met, opening new possibilities for automated business processes.""",
    
    """Cloud computing has democratized access to computational resources, enabling startups and 
    enterprises alike to scale their infrastructure on demand. The shift from capital-intensive 
    on-premises data centers to flexible cloud services has accelerated innovation and reduced 
    barriers to entry for technology companies. Hybrid and multi-cloud strategies are becoming 
    standard as organizations seek to balance performance, cost, and vendor independence.""",
    
    """Cybersecurity has become a critical concern as digital infrastructure becomes more interconnected. 
    Advanced persistent threats, ransomware attacks, and data breaches pose significant risks to 
    organizations and individuals alike. Machine learning is increasingly being applied to detect 
    anomalies and prevent attacks in real-time, while encryption and zero-trust architectures provide 
    fundamental security guarantees for sensitive data and communications.""",
    
    """Internet of Things devices are generating unprecedented amounts of sensor data from physical 
    systems. From smart cities monitoring traffic patterns to industrial IoT optimizing manufacturing 
    processes, the integration of physical and digital worlds creates new opportunities. Edge computing 
    enables processing this data closer to its source, reducing latency and bandwidth requirements 
    while maintaining privacy and enabling real-time decision-making.""",
    
    """Robotics combines mechanical engineering, electronics, and artificial intelligence to create 
    autonomous systems capable of performing complex tasks. From warehouse automation and surgical 
    robots to exploration of hazardous environments, robots are expanding human capabilities. Advanced 
    control systems, sensor fusion, and machine learning enable robots to adapt to dynamic environments 
    and collaborate safely with human workers.""",
    
    """Augmented and virtual reality technologies are transforming how we interact with digital content. 
    AR overlays digital information onto the physical world, enhancing fields from education to 
    industrial maintenance. VR creates fully immersive environments for training, entertainment, and 
    collaboration. As hardware improves and becomes more accessible, these technologies promise to 
    fundamentally change how we work, learn, and communicate.""",
    
    """5G networks provide unprecedented wireless bandwidth and low latency, enabling new applications 
    from autonomous vehicles to remote surgery. The combination of high-speed connectivity, edge 
    computing, and IoT devices creates intelligent infrastructure that can respond in real-time. 
    Network slicing allows customized virtual networks for different use cases, while massive device 
    connectivity supports dense IoT deployments.""",
    
    """Bioinformatics applies computational techniques to understand biological systems and genomic 
    data. Machine learning accelerates drug discovery by predicting molecular interactions and 
    identifying promising compounds. Personalized medicine uses genetic information to tailor treatments 
    to individual patients. CRISPR and gene editing technologies, combined with AI-driven analysis, 
    are revolutionizing healthcare and biotechnology.""",
    
    """Renewable energy systems require sophisticated optimization and forecasting to integrate with 
    power grids. Machine learning predicts solar and wind generation patterns, enabling better energy 
    management. Smart grids use real-time data to balance supply and demand, reducing waste and 
    improving reliability. Energy storage technologies and distributed generation are transforming 
    the power infrastructure toward sustainability.""",
    
    """Autonomous vehicles combine computer vision, sensor fusion, and decision-making algorithms to 
    navigate complex environments. LiDAR, radar, and cameras provide comprehensive environmental 
    awareness, while deep learning processes this data to detect objects and predict behavior. 
    V2X communication enables vehicles to coordinate with infrastructure and each other, improving 
    safety and traffic flow in smart transportation systems.""",
    
    """Digital twins create virtual replicas of physical systems, enabling simulation and optimization 
    without disrupting real operations. From manufacturing plants to entire cities, these models 
    integrate real-time sensor data with physics-based simulations. Predictive maintenance uses 
    digital twins to forecast equipment failures before they occur, reducing downtime and costs 
    across industries.""",
    
    """Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement 
    to perform certain computations exponentially faster than classical computers. While still in 
    early stages, quantum algorithms show promise for cryptography, optimization, and simulating 
    quantum systems. Hybrid quantum-classical approaches are emerging as practical near-term solutions 
    for specific problem domains.""",
    
    """Edge AI brings machine learning inference to edge devices, enabling real-time processing without 
    cloud connectivity. Specialized hardware accelerators and model compression techniques make AI 
    practical on resource-constrained devices. This enables privacy-preserving applications, reduces 
    latency, and decreases bandwidth requirements while maintaining high accuracy for computer vision 
    and language tasks.""",
    
    """Federated learning trains machine learning models across distributed devices while keeping data 
    localized, preserving privacy. This approach enables learning from sensitive data in healthcare, 
    finance, and personal devices without centralizing it. Differential privacy and secure aggregation 
    provide mathematical guarantees that individual data cannot be reconstructed from model updates.""",
    
    """Explainable AI aims to make machine learning models interpretable and transparent. As AI systems 
    make critical decisions in healthcare, finance, and justice, understanding their reasoning becomes 
    essential. Techniques like attention visualization, SHAP values, and counterfactual explanations 
    help humans trust and validate AI predictions, addressing the black-box nature of deep learning.""",
    
    """Graph neural networks process data with complex relational structures, from social networks to 
    molecular compounds. By learning representations that capture both node features and graph topology, 
    GNNs excel at tasks like drug discovery, recommendation systems, and knowledge graph completion. 
    Message passing and attention mechanisms enable information propagation across network structures.""",
    
    """Neuromorphic computing mimics biological neural systems with event-driven, asynchronous processing. 
    Spiking neural networks and specialized hardware offer extreme energy efficiency for certain AI 
    tasks. This brain-inspired approach promises to overcome limitations of traditional von Neumann 
    architectures, particularly for always-on sensor processing and robotics applications.""",
    
    """Synthetic data generation creates artificial training data to augment or replace real datasets. 
    Generative models like GANs and diffusion models produce realistic images, text, and structured 
    data. This addresses data scarcity, privacy concerns, and bias in training sets, while enabling 
    simulation of rare events for testing autonomous systems and other safety-critical applications.""",
    
    """Multi-modal learning combines information from different data types like images, text, and audio. 
    Models like CLIP learn joint representations that enable zero-shot classification and cross-modal 
    retrieval. This approach mirrors human perception and reasoning, enabling richer understanding 
    and more versatile AI systems that can process and relate diverse information sources.""",
    
    """Continual learning enables AI systems to acquire new knowledge without forgetting previous learning. 
    This addresses catastrophic forgetting in neural networks, essential for systems that must adapt 
    to changing environments. Techniques like experience replay, dynamic architectures, and regularization 
    help models maintain performance on old tasks while learning new ones.""",
    
    """Self-supervised learning extracts supervisory signals from data structure itself, reducing 
    dependence on labeled datasets. Contrastive methods and masked prediction pre-train powerful 
    representations from unlabeled data. This paradigm has revolutionized NLP with models like BERT 
    and computer vision with approaches like SimCLR, dramatically reducing annotation requirements.""",
    
    """Neural architecture search automates the design of neural network structures, discovering novel 
    architectures optimized for specific tasks and hardware. While computationally expensive, NAS 
    has found state-of-the-art models for image classification, object detection, and mobile deployment. 
    Efficient NAS methods and transfer of discovered architectures make this increasingly practical.""",
    
    """Compressed sensing and sparse learning recover signals from incomplete measurements by exploiting 
    structure and redundancy. These techniques enable efficient data acquisition in medical imaging, 
    wireless communications, and sensor networks. Deep learning is increasingly combined with compressed 
    sensing principles to learn optimal sampling patterns and reconstruction algorithms.""",
    
    """Causal inference seeks to understand cause-and-effect relationships beyond correlation. Machine 
    learning methods for causal discovery and estimation enable better decision-making by predicting 
    intervention outcomes. This is crucial for scientific discovery, policy evaluation, and AI systems 
    that must reason about actions and consequences in complex environments.""",
    
    """Meta-learning or learning to learn enables models to quickly adapt to new tasks with limited data. 
    By learning across many related tasks, meta-learners extract transferable knowledge about learning 
    itself. Few-shot learning applications range from personalized recommendations to rapid adaptation 
    of robots to new environments, reducing the data and time required for deployment.""",
    
    """Adversarial robustness addresses vulnerability of neural networks to carefully crafted perturbations. 
    Small input changes imperceptible to humans can fool AI systems, raising security concerns. 
    Adversarial training, certified defenses, and robust architectures improve model resilience, 
    essential for deploying AI in safety-critical and adversarial settings like autonomous driving.""",
]

GENERATION_PROMPTS = [
    "Write a detailed explanation of how neural networks learn:",
    "Describe the key differences between supervised and unsupervised learning:",
    "Explain the concept of transfer learning in machine learning:",
    "Discuss the challenges of training large language models:",
    "What are the main applications of reinforcement learning?",
    "Explain how attention mechanisms work in transformers:",
    "Describe the process of fine-tuning a pre-trained model:",
    "What is the significance of gradient descent in deep learning?",
]


async def generate_embedding(client: httpx.AsyncClient, text: str, index: int) -> Dict:
    """Generate embedding for a single text."""
    start = time.time()
    response = await client.post(
        f"{BASE_URL}/embeddings",
        json={"input": text, "model": EMBEDDING_MODEL},
        timeout=60.0,
    )
    duration = time.time() - start
    response.raise_for_status()
    return {"index": index, "duration": duration, "type": "embedding"}


async def generate_completion(client: httpx.AsyncClient, prompt: str, index: int) -> Dict:
    """Generate completion for a single prompt."""
    start = time.time()
    response = await client.post(
        f"{BASE_URL}/completions",
        json={
            "prompt": prompt,
            "model": GENERATION_MODEL,
            "max_tokens": 100,  # Increased for more balanced timing
            "temperature": 0.7,
        },
        timeout=60.0,
    )
    duration = time.time() - start
    response.raise_for_status()
    return {"index": index, "duration": duration, "type": "completion"}


async def run_sequential(client: httpx.AsyncClient) -> Dict:
    """Sequential execution: embeddings first, then completions."""
    print("\n[1/2] Sequential Execution (Traditional Approach)")
    print("      Run embeddings, THEN completions...")
    
    start_time = time.time()
    
    # Phase 1: Embeddings
    embedding_start = time.time()
    embedding_tasks = [
        generate_embedding(client, text, i) 
        for i, text in enumerate(EMBEDDING_TEXTS)
    ]
    embedding_results = await asyncio.gather(*embedding_tasks)
    embedding_duration = time.time() - embedding_start
    avg_embed_time = sum(r["duration"] for r in embedding_results) / len(embedding_results)
    
    # Phase 2: Completions
    completion_start = time.time()
    completion_tasks = [
        generate_completion(client, prompt, i) 
        for i, prompt in enumerate(GENERATION_PROMPTS)
    ]
    completion_results = await asyncio.gather(*completion_tasks)
    completion_duration = time.time() - completion_start
    avg_comp_time = sum(r["duration"] for r in completion_results) / len(completion_results)
    
    total_duration = time.time() - start_time
    print(f"      ✓ Done in {total_duration:.2f}s (embed: {embedding_duration:.2f}s, gen: {completion_duration:.2f}s)")
    
    return {
        "mode": "sequential",
        "embedding_duration": embedding_duration,
        "completion_duration": completion_duration,
        "total_duration": total_duration,
        "embeddings_count": len(embedding_results),
        "completions_count": len(completion_results),
        "avg_embedding_time": avg_embed_time,
        "avg_completion_time": avg_comp_time,
    }


async def run_concurrent(client: httpx.AsyncClient) -> Dict:
    """Concurrent execution: embeddings and completions simultaneously."""
    print("\n[2/2] Concurrent Execution (memoria_vllm)")
    print("      Run embeddings AND completions simultaneously...")
    
    start_time = time.time()
    
    # Create all tasks
    embedding_tasks = [
        generate_embedding(client, text, i) 
        for i, text in enumerate(EMBEDDING_TEXTS)
    ]
    completion_tasks = [
        generate_completion(client, prompt, i) 
        for i, prompt in enumerate(GENERATION_PROMPTS)
    ]
    
    # Run everything concurrently
    all_results = await asyncio.gather(*(embedding_tasks + completion_tasks))
    
    total_duration = time.time() - start_time
    
    # Separate results by type
    embedding_results = [r for r in all_results if r["type"] == "embedding"]
    completion_results = [r for r in all_results if r["type"] == "completion"]
    
    avg_embed_time = sum(r["duration"] for r in embedding_results) / len(embedding_results)
    avg_comp_time = sum(r["duration"] for r in completion_results) / len(completion_results)
    
    print(f"      ✓ Done in {total_duration:.2f}s (both models running on ONE GPU)")
    
    return {
        "mode": "concurrent",
        "total_duration": total_duration,
        "embeddings_count": len(embedding_results),
        "completions_count": len(completion_results),
        "avg_embedding_time": avg_embed_time,
        "avg_completion_time": avg_comp_time,
    }


def print_comparison(sequential_result: Dict, concurrent_result: Dict):
    """Print concise comparison table."""
    seq_total = sequential_result["total_duration"]
    conc_total = concurrent_result["total_duration"]
    seq_embed = sequential_result["embedding_duration"]
    seq_comp = sequential_result["completion_duration"]
    speedup = seq_total / conc_total
    time_saved = seq_total - conc_total
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Results table
    print(f"\n{'Metric':<30} {'Sequential':<15} {'Concurrent':<15}")
    print("-" * 70)
    print(f"{'Embedding requests':<30} {sequential_result['embeddings_count']:<15} {concurrent_result['embeddings_count']:<15}")
    print(f"{'Completion requests':<30} {sequential_result['completions_count']:<15} {concurrent_result['completions_count']:<15}")
    print(f"{'Total time (s)':<30} {seq_total:<15.2f} {conc_total:<15.2f}")
    print(f"{'Embedding phase (s)':<30} {seq_embed:<15.2f} {'N/A':<15}")
    print(f"{'Completion phase (s)':<30} {seq_comp:<15.2f} {'N/A':<15}")
    print("-" * 70)
    print(f"{'Speedup':<30} {'-':<15} {speedup:.2f}x")
    print(f"{'Time saved (s)':<30} {'-':<15} {time_saved:.2f}")
    print(f"{'Efficiency gain':<30} {'-':<15} {(time_saved/seq_total)*100:.1f}%")
    
    print("\n" + "="*70)
    print(f"💡 memoria_vllm runs both models simultaneously on ONE GPU")
    print(f"   Alternatives: 2 GPUs ($2000+) OR model swapping (slow)")
    print("="*70)


async def check_health():
    """Verify all services are healthy."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://localhost:8200/health", timeout=10.0)
            response.raise_for_status()
            health = response.json()
            if health["status"] == "healthy":
                return True
            else:
                print("✗ Services not healthy:", health)
                return False
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            print("Start services: sudo docker compose up -d")
            return False


async def main():
    """Run the balanced performance demo."""
    print("="*70)
    print("memoria_vllm: Multi-Model Concurrent Performance Demo")
    print("="*70)
    print(f"Workload: {len(EMBEDDING_TEXTS)} embeddings + {len(GENERATION_PROMPTS)} completions")
    print(f"Hardware: Single GPU shared by both models")
    
    # Check health
    if not await check_health():
        return
    
    # Run tests
    async with httpx.AsyncClient() as client:
        sequential_result = await run_sequential(client)
        await asyncio.sleep(2)
        concurrent_result = await run_concurrent(client)
    
    # Compare results
    print_comparison(sequential_result, concurrent_result)


if __name__ == "__main__":
    asyncio.run(main())
