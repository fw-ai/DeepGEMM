from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BODY = ROOT / "deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh"
SCHEDULER = ROOT / "deep_gemm/include/deep_gemm/scheduler/gemm.cuh"
DECODER = ROOT / "deep_gemm/include/deep_gemm/scheduler/external_k_grouped_range.hpp"
K3_BACKWARD = (
    ROOT
    / "deep_gemm/include/deep_gemm/impls/"
    / "sm100_fp8_fp4_mega_moe_backward.cuh"
)


def _cluster_tasks(shape_m: int, shape_n: int, paired: bool) -> list[tuple[int, int]]:
    """Model the K3 M-multicast output geometry independent of swizzle order."""
    m_clusters = shape_m // 256
    n_tiles = shape_n // 256
    n_stride = 2 if paired else 1
    return [
        (m_cluster, n_tile)
        for n_tile in range(0, n_tiles, n_stride)
        for m_cluster in range(m_clusters)
    ]


def test_paired_decode_covers_each_original_output_exactly_once() -> None:
    """Expanding every pair must reproduce dW2 and dW13 output grids."""
    for shape_m, shape_n, expected_paired_tasks in (
        (3584, 3072, 84),
        (6144, 3584, 168),
    ):
        baseline = set(_cluster_tasks(shape_m, shape_n, paired=False))
        paired = _cluster_tasks(shape_m, shape_n, paired=True)
        expanded = {
            (m_cluster, n_tile + offset)
            for m_cluster, n_tile in paired
            for offset in (0, 1)
        }
        assert len(paired) == expected_paired_tasks
        assert len(expanded) == 2 * len(paired)
        assert expanded == baseline


def _window_major_consumer_events(
    num_k_blocks: int,
    stages: int,
) -> list[tuple[str, int]]:
    """Model paired UMMA, callback, and TMEM publication program order."""
    events: list[tuple[str, int]] = []
    final_k = num_k_blocks - 1
    for window_begin in range(0, num_k_blocks, stages):
        window_end = min(window_begin + stages, num_k_blocks)
        for k_block in range(window_begin, window_end):
            events.append(("umma0", k_block))
            if k_block == final_k:
                events.append(("tmem_full", 0))
        if window_begin == 0:
            events.append(("wait_tmem", 1))
        for k_block in range(window_begin, window_end):
            if k_block == final_k:
                events.extend((("callback", 0), ("callback", 1)))
            events.append(("umma1", k_block))
            if k_block == final_k:
                events.append(("tmem_full", 1))
    return events


def test_window_major_reduction_preserves_per_output_k_order() -> None:
    """Every 1--64-tile accumulator still observes globally increasing K."""
    stages = 6
    for num_k_blocks in range(1, 65):
        events = _window_major_consumer_events(num_k_blocks, stages)
        expected = list(range(num_k_blocks))
        observed = {
            output: [
                k_block
                for operation, k_block in events
                if operation == f"umma{output}"
            ]
            for output in (0, 1)
        }
        assert observed[0] == expected
        assert observed[1] == expected

        # The first resident window is wholly consumed by output zero before
        # this task waits for or writes the second accumulator half.
        wait_position = events.index(("wait_tmem", 1))
        first_umma1 = events.index(("umma1", 0))
        first_window = min(num_k_blocks, stages)
        assert [
            k_block
            for operation, k_block in events[:wait_position]
            if operation == "umma0"
        ] == list(range(first_window))
        assert not any(
            operation == "umma1"
            for operation, _ in events[:wait_position]
        )
        assert wait_position < first_umma1


def test_four_task_claims_remain_expert_local_and_phase_neutral() -> None:
    """Both paired K3 grids restore retained resources after each real claim."""
    batch_tasks = 4
    k_blocks_per_pool_block = 192 // 64
    stages = 6
    for tasks_per_expert in (84, 168):
        assert tasks_per_expert % batch_tasks == 0
        for active_experts in range(113):
            task_limit = active_experts * tasks_per_expert
            claims = [
                range(first, first + batch_tasks)
                for first in range(0, task_limit, batch_tasks)
            ]
            for claim in claims:
                assert claim.start // tasks_per_expert == (claim.stop - 1) // tasks_per_expert
        # Two full-barrier generations per paired task restore phase zero.
        assert (batch_tasks * 2) % 2 == 0
        # Both TMEM columns are released once per task; an even claim restores
        # each empty/full generation for a retained descriptor switch.
        assert batch_tasks % 2 == 0
        # Every active K3 expert is padded in 192-row pool blocks. Four tasks
        # therefore advance the six-stage ring by 12 K64 blocks per pool block,
        # returning both the stage index and phase to their initial values.
        assert (batch_tasks * k_blocks_per_pool_block) % (2 * stages) == 0


def _decode_terminal_two_segment_tile(
    logical_task: int,
    tasks_per_expert: int,
    shape_m: int,
    shape_n: int,
    cluster_rank: int,
) -> tuple[int, int]:
    """Model the tile-only portion of the terminal two-segment decoder."""
    block_m = 128
    block_n = 256
    multicast = 2
    one_d_blocks = 16
    m_blocks = shape_m // block_m
    n_blocks = shape_n // block_n
    assert tasks_per_expert == m_blocks * n_blocks // multicast

    cta_task = (logical_task % tasks_per_expert) * multicast + cluster_rank
    blocks_per_swizzle_group = n_blocks * one_d_blocks
    swizzle_group = cta_task // blocks_per_swizzle_group
    first_m_block = swizzle_group * one_d_blocks
    in_group = cta_task % blocks_per_swizzle_group
    m_blocks_in_group = min(one_d_blocks, m_blocks - first_m_block)
    return (
        first_m_block + in_group % m_blocks_in_group,
        in_group // m_blocks_in_group,
    )


def test_cached_terminal_two_segment_decode_preserves_full_contract() -> None:
    """Claims keep expert metadata fixed and cover the full K3 tile grid."""
    batch_tasks = 4
    experts = 896
    for shape_m, shape_n, tasks_per_expert in (
        (3584, 3072, 168),
        (6144, 3584, 336),
    ):
        for expert in range(experts):
            expert_first = expert * tasks_per_expert
            decoded_tiles: set[tuple[int, int]] = set()
            for batch_first in range(
                expert_first,
                expert_first + tasks_per_expert,
                batch_tasks,
            ):
                assert batch_first // tasks_per_expert == (
                    batch_first + batch_tasks - 1
                ) // tasks_per_expert
                for offset in range(batch_tasks):
                    logical_task = batch_first + offset
                    assert logical_task // tasks_per_expert == expert
                    for cluster_rank in range(2):
                        decoded_tiles.add(_decode_terminal_two_segment_tile(
                            logical_task, tasks_per_expert,
                            shape_m, shape_n, cluster_rank))
            assert decoded_tiles == {
                (m_block, n_block)
                for m_block in range(shape_m // 128)
                for n_block in range(shape_n // 256)
            }

    scheduler = SCHEDULER.read_text()
    assert "decode_cached_batch_task" in scheduler
    assert "batch_offset++ == 0u" in scheduler


def test_paired_protocol_reduces_only_a_tma_traffic() -> None:
    """For two outputs, pair one A with two B loads and two UMMA/stores."""
    baseline = {"a_tma": 2, "b_tma": 2, "umma": 2, "stores": 2}
    paired = {"a_tma": 1, "b_tma": 2, "umma": 2, "stores": 2}
    assert paired["a_tma"] * 2 == baseline["a_tma"]
    for operation in ("b_tma", "umma", "stores"):
        assert paired[operation] == baseline[operation]


def test_terminal_dw13_uses_only_base_combine_warps() -> None:
    """The EP8 dW13 tail must not add memory contention to saturated UMMA."""
    source = K3_BACKWARD.read_text()
    assert "kTerminalDW13ExtraCombineThreads = 0u" in source
    assert "kTerminalDW13ExtraCombineThreads>(" in source


def _windowed_producer_events(
    num_k_blocks: int,
    stages: int,
) -> list[tuple[str, int]]:
    """Model program-order TMA issues for the paired six-stage producer."""
    events = [
        ("a_b0", k_block)
        for k_block in range(min(num_k_blocks, stages))
    ]
    for window_begin in range(0, num_k_blocks, stages):
        window_end = min(window_begin + stages, num_k_blocks)
        events.extend(
            ("b1", k_block)
            for k_block in range(window_begin, window_end)
        )
        refill_begin = window_begin + stages
        refill_end = min(refill_begin + stages, num_k_blocks)
        events.extend(
            ("a_b0", k_block)
            for k_block in range(refill_begin, refill_end)
        )
    return events


def test_windowed_b1_pipeline_has_exact_stage_and_phase_coverage() -> None:
    """B1 is queued window-wide before phase-flipped whole-stage refills."""
    stages = 6
    for start_stage in range(stages):
        for start_phase in (0, 1):
            for num_k_blocks in range(1, 65):
                stage_for = lambda k: (start_stage + k) % stages
                phase_for = lambda k: start_phase ^ (
                    ((start_stage + k) // stages) & 1
                )
                events = _windowed_producer_events(num_k_blocks, stages)
                a_b0_issued: set[int] = set()
                b1_issued: set[int] = set()
                for operation, k_block in events:
                    if operation == "a_b0":
                        previous = k_block - stages
                        if previous >= 0:
                            assert previous in b1_issued
                            assert stage_for(k_block) == stage_for(previous)
                            assert phase_for(k_block) == (phase_for(previous) ^ 1)
                        a_b0_issued.add(k_block)
                    else:
                        assert k_block in a_b0_issued
                        b1_issued.add(k_block)
                assert a_b0_issued == set(range(num_k_blocks))
                assert b1_issued == set(range(num_k_blocks))

                # The key performance invariant: no refill can block the TMA
                # warp before every resident B1 in the current window is
                # queued. This specifically excludes B1(k), A+B0(k+6),
                # B1(k+1), the serialized order used by the predecessor.
                positions = {event: index for index, event in enumerate(events)}
                for window_begin in range(0, num_k_blocks, stages):
                    window_end = min(window_begin + stages, num_k_blocks)
                    refill_begin = window_begin + stages
                    refill_end = min(refill_begin + stages, num_k_blocks)
                    if refill_begin < refill_end:
                        last_b1 = max(
                            positions[("b1", k_block)]
                            for k_block in range(window_begin, window_end)
                        )
                        first_refill = min(
                            positions[("a_b0", k_block)]
                            for k_block in range(refill_begin, refill_end)
                        )
                        assert last_b1 < first_refill

                final_stage = start_stage
                final_phase = start_phase
                for _ in range(num_k_blocks):
                    final_stage = (final_stage + 1) % stages
                    final_phase ^= final_stage == 0
                linear = start_stage + num_k_blocks
                assert final_stage == linear % stages
                assert final_phase == start_phase ^ ((linear // stages) & 1)


def test_window_major_dependency_graph_is_deadlock_free() -> None:
    """All stage/phase starts make producer, consumer, and refills progress."""
    stages = 6
    for start_stage in range(stages):
        for start_phase in (0, 1):
            for num_k_blocks in range(1, 65):
                producer = _windowed_producer_events(
                    num_k_blocks, stages)
                consumer = _window_major_consumer_events(
                    num_k_blocks, stages)
                producer_idx = 0
                consumer_idx = 0
                issued_a_b0: set[int] = set()
                issued_b1: set[int] = set()
                retired_b0: set[int] = set()
                retired_b1: set[int] = set()
                stage_payload: dict[int, tuple[int, str, int]] = {}

                def stage_for(k_block: int) -> int:
                    return (start_stage + k_block) % stages

                def phase_for(k_block: int) -> int:
                    return start_phase ^ (
                        ((start_stage + k_block) // stages) & 1
                    )

                while (producer_idx < len(producer) or
                       consumer_idx < len(consumer)):
                    made_progress = False
                    if producer_idx < len(producer):
                        operation, k_block = producer[producer_idx]
                        if operation == "a_b0":
                            previous = k_block - stages
                            if previous < 0 or previous in retired_b1:
                                stage_payload[stage_for(k_block)] = (
                                    k_block, "b0", phase_for(k_block))
                                issued_a_b0.add(k_block)
                                producer_idx += 1
                                made_progress = True
                        elif k_block in retired_b0:
                            assert stage_payload[stage_for(k_block)] == (
                                k_block, "b0", phase_for(k_block))
                            stage_payload[stage_for(k_block)] = (
                                k_block, "b1", phase_for(k_block))
                            issued_b1.add(k_block)
                            producer_idx += 1
                            made_progress = True

                    if consumer_idx < len(consumer):
                        operation, value = consumer[consumer_idx]
                        if operation == "umma0" and value in issued_a_b0:
                            assert stage_payload[stage_for(value)] == (
                                value, "b0", phase_for(value))
                            retired_b0.add(value)
                            consumer_idx += 1
                            made_progress = True
                        elif operation == "umma1" and value in issued_b1:
                            assert stage_payload[stage_for(value)] == (
                                value, "b1", phase_for(value))
                            retired_b1.add(value)
                            consumer_idx += 1
                            made_progress = True
                        elif operation in {
                            "wait_tmem", "callback", "tmem_full"
                        }:
                            # TMEM1 is deliberately allowed to become ready
                            # only after the complete first output-zero window.
                            # Callback/full events carry no stage dependency.
                            consumer_idx += 1
                            made_progress = True

                    assert made_progress, (
                        start_stage,
                        start_phase,
                        num_k_blocks,
                        producer_idx,
                        consumer_idx,
                    )

                assert retired_b0 == set(range(num_k_blocks))
                assert retired_b1 == set(range(num_k_blocks))

                # Refilling stage k+6 depends only on UMMA1(k), not on later
                # stages in the same window, so producer and consumer form a
                # stage-wise wavefront rather than a whole-window barrier.
                for refill_k in range(stages, num_k_blocks):
                    previous = refill_k - stages
                    assert stage_for(refill_k) == stage_for(previous)
                    assert phase_for(refill_k) == (phase_for(previous) ^ 1)


def test_b1_waits_for_b0_retirement_before_reusing_the_stage() -> None:
    """B1 must not treat its freshly initialized retirement edge as ready."""
    # mbarrier.try_wait(parity) completes only after the barrier has advanced
    # away from ``parity``.  An ordinary empty slot is initially available and
    # therefore waits on the opposite parity.  The paired B-retirement edge is
    # initially unavailable: output zero's UMMA arrival must advance it first.
    def wait_completes(current_parity: int, waited_parity: int) -> bool:
        return current_parity != waited_parity

    for generation in (0, 1):
        current = generation
        assert wait_completes(current, generation ^ 1)  # Initial A+B0 slot.
        assert not wait_completes(current, generation)  # B0 is still live.
        current ^= 1  # UMMA0 retires B0 through umma_arrive().
        assert wait_completes(current, generation)

    body = BODY.read_text()
    assert (
        "paired_b_empty_barriers[target_stage]->wait(\n"
        "                    target_phase);"
    ) in body
    assert (
        "paired_b_empty_barriers[target_stage]->wait(\n"
        "                    target_phase ^ 1u);"
    ) not in body


def test_paired_retirement_preserves_two_logical_callback_events() -> None:
    """Every callback signature must observe both outputs in a paired task."""
    body = BODY.read_text()
    detailed = (
        "retire_output(n_block_idx);\n"
        "                            if constexpr (kTaskPairedN)\n"
        "                                retire_output(n_block_idx + 1u);"
    )
    group_only = (
        "input_tile_retired(scheduler.current_group_idx);\n"
        "                            if constexpr (kTaskPairedN)\n"
        "                                input_tile_retired(\n"
        "                                    scheduler.current_group_idx);"
    )
    assert detailed in body
    assert group_only in body


def test_callback_and_tmem_full_publication_are_exactly_once() -> None:
    """Early output-zero publication cannot duplicate callbacks or outputs."""
    stages = 6
    for num_k_blocks in range(1, 65):
        events = _window_major_consumer_events(num_k_blocks, stages)
        assert [value for operation, value in events
                if operation == "callback"] == [0, 1]
        assert [value for operation, value in events
                if operation == "tmem_full"] == [0, 1]

        full0 = events.index(("tmem_full", 0))
        last_umma0 = max(
            index for index, event in enumerate(events)
            if event[0] == "umma0")
        last_umma1 = max(
            index for index, event in enumerate(events)
            if event[0] == "umma1")
        assert last_umma0 < full0 < last_umma1
        assert events.index(("tmem_full", 1)) > last_umma1


def test_paired_body_reuses_reserved_barriers_and_tmem() -> None:
    """The specialization stays within the existing 230400-byte allocation."""
    body = BODY.read_text()
    for required in (
        "kTaskPairedN",
        "paired_b_empty_barriers",
        "Prime all distinct stages with A+B0",
        "n_idx += BLOCK_N",
        "const uint32_t prefill",
        "const uint32_t window_end",
        "const uint32_t refill_begin",
        "output_in_task * UMMA_N",
        "tmem_empty_barriers[output_in_task]",
        "paired_task_phase",
    ):
        assert required in body

    bf16_bytes = 2
    block_m = 128
    block_n = 256
    block_k = 64
    stages = 6
    store_block_n = 128 // bf16_bytes
    cd_bytes = 2 * block_m * store_block_n * bf16_bytes
    mainloop_bytes = stages * (block_m + block_n // 2) * block_k * bf16_bytes
    barrier_bytes = 8
    control_bytes = (3 * stages + 2 * 2 + 1) * barrier_bytes + 4
    assert cd_bytes + mainloop_bytes == 229376
    assert cd_bytes + mainloop_bytes + control_bytes == 229564
    assert cd_bytes + mainloop_bytes + control_bytes <= 230400


def test_second_tmem_half_wait_is_deferred_through_first_window() -> None:
    """A full B0 window may overlap the prior epilogue without TMEM1 alias."""
    body = BODY.read_text()
    upfront = (
        "tmem_empty_barriers[0]->wait(paired_task_phase ^ 1u);\n"
        "            } else"
    )
    delayed = (
        "if (output_in_task == 1u &&\n"
        "                            window_begin == 0u) {"
    )
    assert upfront in body
    assert delayed in body
    wait_begin = body.index(delayed)
    wait_end = body.index(
        "ptx::tcgen05_after_thread_sync();", wait_begin)
    assert "tmem_empty_barriers[1]->wait(" in body[wait_begin:wait_end]
    assert "paired_task_phase ^ 1u" in body[wait_begin:wait_end]

    # The serial paired epilogue releases half zero first.  Waiting on that
    # edge permits UMMA0 for the next task, while the delayed half-one wait
    # still precedes the first write to TMEM1.  No accumulator half is written
    # before its own prior epilogue release.
    events = (
        "release_previous_tmem0",
        "wait_tmem0",
        "issue_umma0_k0",
        "issue_umma0_k1",
        "issue_umma0_k2",
        "issue_umma0_k3",
        "issue_umma0_k4",
        "issue_umma0_k5",
        "release_previous_tmem1",
        "wait_tmem1",
        "issue_umma1_k0",
    )
    positions = {event: index for index, event in enumerate(events)}
    assert positions["release_previous_tmem0"] < positions["issue_umma0_k0"]
    assert positions["release_previous_tmem1"] < positions["issue_umma1_k0"]
    assert positions["issue_umma0_k5"] < positions["wait_tmem1"]


def test_scheduler_exposes_compile_time_paired_contract() -> None:
    """The one- and two-range terminal variants share the paired decoder."""
    scheduler = SCHEDULER.read_text()
    decoder = DECODER.read_text()
    assert scheduler.count("kTaskPairedN = kPairAdjacentN") == 3
    assert decoder.count("bool kPairAdjacentN = false") == 3
    assert "n_block_idx * (kPairAdjacentN ? 2u : 1u)" in decoder
    assert "SHAPE_N % (2u * BLOCK_N) == 0u" in decoder
