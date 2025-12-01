import logging

import pytest

from custom_components.ai_memory.constants import ENGINE_REMOTE, MEMORY_LIMIT, \
    SIMILARITY_THRESHOLD, DEFAULT_REMOTE_URL
from custom_components.ai_memory.embedding_remote import RemoteEmbeddingEngine
from custom_components.ai_memory.memory_manager import MemoryManager

_LOGGER = logging.getLogger(__name__)

MEMORIES = [
    ("The user's name is Arda Güler.", "private"),
    ("The user likes pizza.", "private"),
    ("The user does not want the blinds opened before 8 AM.", "common"),
    ("The user supports Real Madrid.", "private"),
]

QUERIES = [
    {"query": "What football club do I support?", "expect": "Madrid"},
    {"query": "What foods do I likes?", "expect": "pizza"},
    {"query": "Do I like jazz music?", "expect": None},
    {"query": "Hangi takımı tutuyorum?", "expect": "Madrid"},
    {"query": "Hangi yemeği severim?", "expect": "pizza"},
    {"query": "Jazz müzik sever miyim?", "expect": None},
]


@pytest.fixture(scope="function")
async def ollama_available():
    """Check if Ollama is running on localhost."""
    engine = RemoteEmbeddingEngine(None, {"remote_url": DEFAULT_REMOTE_URL})
    is_available = await engine.async_get_version()
    print(is_available)
    if not is_available:
        pytest.skip(f"Ollama not available at {DEFAULT_REMOTE_URL} - skipping benchmark tests")
    return is_available


@pytest.mark.asyncio
class TestMemoryBenchmark:

    async def test_memory_retrieval_scenarios(self, hass, ollama_available):
        _LOGGER.info("\n--- HA-AI-MEMORY PYTEST BENCHMARK START ---")

        manager = MemoryManager(hass, ENGINE_REMOTE)

        _LOGGER.info(f"\n1. Injecting {len(MEMORIES)} memories into virtual DB...")
        for text, scope in MEMORIES:
            await manager.async_add_memory(text, scope, "pytest_agent")
        _LOGGER.info("Done injection.")

        _LOGGER.info("\n2. Running Query Scenarios...")
        _LOGGER.info(f"{'QUERY':<35} | {'TOP RESULT':<30} | {'SCORE':<5} | {'STATUS'}")
        _LOGGER.info("-" * 120)

        total_passed = 0

        for case in QUERIES:
            query = case["query"]
            expected_keyword = case["expect"]

            results = await manager.async_search_memory(query, "pytest_agent", limit=MEMORY_LIMIT,
                                                        min_score=SIMILARITY_THRESHOLD)

            if not results:
                top_text = "NONE (Empty List)"
                score = 0.0
                if expected_keyword is None:
                    status = "✅ PASS (Noise Filtered)"
                    is_passed = True
                else:
                    status = "❌ FAIL (Missed Data)"
                    is_passed = False
            else:
                top_match = results[0]
                top_text = top_match['content']
                score = top_match['score']

                if expected_keyword is None:
                    status = "❌ FAIL (Hallucination)"
                    is_passed = False
                elif expected_keyword.lower() in top_match['content'].lower():
                    status = "✅ PASS (Found)"
                    is_passed = True
                else:
                    status = "⚠️ WEAK (Wrong Context)"
                    is_passed = False

            _LOGGER.info(f"{query:<35} | {top_text:<30} | {score:.3f} | {status}")

            if is_passed:
                total_passed += 1

        _LOGGER.info("-" * 120)
        _LOGGER.info(f"Total Tests: {len(QUERIES)}, Passed: {total_passed}")

        assert total_passed == len(QUERIES), f"Only {total_passed}/{len(QUERIES)} queries passed the benchmark."
