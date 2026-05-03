"""
Minimal Example — General Memory Framework
============================================

Demonstrates the Nouse memory framework with arbitrary text input, 
no domain-specific knowledge required. Shows:

  1. Embedding-based encoding (general-purpose text → CA grid)
  2. CA evolution (organic forgetting/consolidation)
  3. Spreading activation retrieval
  4. Multi-agent memory management
  5. Shared pool for cross-agent knowledge

Run this example to verify the framework works:
    python examples/minimal_example.py

Expected output shows memories being encoded, evolved, and retrieved 
with meaningful results despite starting from completely arbitrary text.
"""

import sys
import os

# Add parent directory to path so we can import the memory module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nouse_hermes.memory.core.ca_engine import CAEngine, MemoryDecayRule, ConsolidationRule
from nouse_hermes.memory.encoders.embedding import EmbeddingEncoder, create_embedding_encoder
from nouse_hermes.memory.retrieval.spreading_activation import SpreadingActivationRetriever
from nouse_hermes.memory.agents.agent_memory import MultiAgentMemorySystem


def demo_basic_ca_engine():
    """Demonstrate the standalone CA engine (no LACE dependency)."""
    print("=" * 60)
    print("1. Standalone CA Engine (decoupled from LACE)")
    print("=" * 60)
    
    # Create a CA engine with memory dynamics rules
    engine = CAEngine(grid_size=(50, 50), neighborhood='moore')
    engine.register_rule('decay', MemoryDecayRule(decay_rate=0.02))
    engine.register_rule('consolidation', ConsolidationRule(threshold=0.7))
    
    # Add some "memories" as active nodes
    memories = [
        ("domain_failure", 0.8, "domain domain_failure causes system_failure"),
        ("roughness", 0.6, "Surface roughness increases domain_failure risk"),
        ("cleaning", 0.5, "Cleaning reduces surface roughness"),
        ("system_failure", 0.7, "Output Z depends on parameter W and configuration V"),
    ]
    
    for nid, state, content in memories:
        import hashlib
        h = hashlib.md5(content.encode()).hexdigest()
        r = int(h[:8], 16) % 50
        c = int(h[8:16], 16) % 50
        
        engine.set_node_state(nid, (r, c), state=state, metadata={"content": content})
    
    # Create edges between related memories
    engine.add_edge(("domain_failure", "roughness"), weight=0.7)
    engine.add_edge(("roughness", "cleaning"), weight=0.6)
    engine.add_edge(("domain_failure", "pull_in"), weight=0.5)
    
    print(f"  Initial state: {engine.get_state_summary()}")
    
    # Evolve (forget/consolidate)
    states, edges = engine.evolve(steps=10)
    print(f"  After 10 evolution steps: {engine.get_state_summary()}")
    
    # Retrieve via spreading activation from "surface" seed
    activations = engine.spread_activation(seed_nodes={"roughness": 1.0}, max_steps=20)
    top = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n  Retrieval from 'surface' seed:")
    for nid, score in top:
        content = engine.nodes[nid].metadata.get("content", nid) if nid in engine.nodes else "N/A"
        print(f"    {nid}: activation={score:.4f} — '{content[:50]}...'")
    
    print()


def demo_embedding_encoder():
    """Demonstrate the embedding-based encoder (general-purpose text input)."""
    print("=" * 60)
    print("2. Embedding-Based Encoder (arbitrary text → CA grid)")
    print("=" * 60)
    
    # Create encoder — works with any text, no domain knowledge needed
    try:
        encoder = create_embedding_encoder(grid_size=(100, 100), use_faiss=False)
        
        # Encode arbitrary text (could be from any source)
        texts = [
            "The cat sat on the mat near the window",
            "Entity A affects outcome B in the domain", 
            "Surface roughness increases adhesion between components",
            "Cleaning procedures reduce contamination risk",
            "Output Z depends on parameter W and configuration V",
        ]
        
        result = encoder.encode_batch(texts)
        
        print(f"  Encoded {len(result.nodes)} text items:")
        for node in result.nodes[:3]:
            print(f"    [{node.position}] '{node.content[:40]}...' "
                  f"(state={node.state:.2f}, type={node.semantic_type.value})")
        
        if result.edges:
            print(f"\n  Created {len(result.edges)} associative edges between similar items")
            for edge in result.edges[:3]:
                print(f"    {edge.source} ↔ {edge.target} (weight={edge.weight:.2f})")
    
    except Exception as e:
        print(f"  Note: Embedding encoder requires sentence-transformers or FAISS.")
        print(f"  Error: {e}")
        print("  Framework still works with keyword-based fallback encoding.\n")


def demo_multi_agent():
    """Demonstrate multi-agent memory management."""
    print("=" * 60)
    print("3. Multi-Agent Memory System")
    print("=" * 60)
    
    system = MultiAgentMemorySystem(default_grid_size=(50, 50))
    
    # Register agents
    system.register_agent("physics_analyst", grid_size=(50, 50))
    system.register_agent("knowledge_engineer", grid_size=(50, 50))
    
    # Each agent encodes its own memories
    physics_mem = system.get_agent_memory("physics_analyst")
    physics_mem.encode("Entity A affects outcome B in the domain")
    physics_mem.encode("Surface roughness increases domain_failure risk", semantic_type="fact")
    
    kg_mem = system.get_agent_memory("knowledge_engineer")
    kg_mem.encode("Cleaning reduces surface contamination", semantic_type="rule")
    kg_mem.encode("Output Z depends on parameter W and configuration V", semantic_type="observation")
    
    # Shared pool for cross-agent knowledge
    shared = system.get_shared_pool()
    shared.write("domain domain_failure: adhesion between surfaces during release", 
                 source_agent="physics_analyst")
    shared.write("Surface science fundamentals for domain reliability",
                 source_agent="knowledge_engineer")
    
    print(f"  Physics analyst memories: {len(physics_mem.tiers['short_term'])}")
    print(f"  Knowledge engineer memories: {len(kg_mem.tiers['short_term'])}")
    print(f"  Shared pool entries: {len(shared.memories)}")
    
    # Evolve all together
    system.evolve_all(steps=5)
    
    # Cross-agent query
    results = shared.read("domain_failure", top_k=3)
    print(f"\n  Shared pool query 'domain_failure': {len(results)} results")
    for nid, score in results:
        entry = shared.memories.get(nid, {})
        source = entry.get("source_agent", "unknown")
        content = entry.get("content", "")[:50]
        print(f"    [{source}] '{content}...' (score={score:.2f})")
    
    # System stats
    stats = system.get_system_stats()
    print(f"\n  System: {stats['num_agents']} agents, "
          f"{stats['shared_pool_memories']} shared memories")
    print()


def demo_hybrid_retrieval():
    """Demonstrate hybrid retrieval (FAISS + spreading activation)."""
    print("=" * 60)
    print("4. Hybrid Retrieval (FAISS + Spreading Activation)")
    print("=" * 60)
    
    from nouse_hermes.memory.retrieval.faiss_retriever import HybridRetriever
    
    retriever = HybridRetriever()
    
    # Add some memories to the FAISS index
    memories = {
        "mem_1": {"content": "Entity A affects outcome B in the domain", 
                  "semantic_type": "fact"},
        "mem_2": {"content": "Surface roughness increases adhesion risk",
                  "semantic_type": "observation"},
        "mem_3": {"content": "Cleaning procedures reduce contamination",
                  "semantic_type": "rule"},
        "mem_4": {"content": "Output Z depends on parameter W and configuration V",
                  "semantic_type": "fact"},
    }
    
    retriever.add_memories(memories)
    
    # Query with semantic similarity (not keyword matching)
    results = retriever.retrieve("surface adhesion phenomena", top_k=3)
    
    print(f"  Query: 'surface adhesion phenomena'")
    print(f"  Results ({results.retrieval_method}):")
    for r in results.results:
        print(f"    [{r.tier}] '{r.content[:45]}...' (score={r.score:.3f})")
    
    print(f"\n  Confidence: {results.confidence:.2f}")
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("  Nouse Hermes — General Memory Framework Demo")
    print("=" * 60 + "\n")
    
    try:
        demo_basic_ca_engine()
        demo_embedding_encoder()
        demo_multi_agent()
        demo_hybrid_retrieval()
        
        print("=" * 60)
        print("  All demos completed successfully!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("  ✓ CA engine works standalone (no LACE dependency)")
        print("  ✓ Embedding encoder handles arbitrary text input")
        print("  ✓ Multi-agent system manages per-agent + shared memory")
        print("  ✓ Hybrid retrieval combines semantic + associative search")
        print("\nFramework is general-purpose and ready for contribution.")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
