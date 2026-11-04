import os

import pytest

from arx_x5_data.jobs import physical_cpu_groups, queue_paths
from arx_x5_data.validation import validate_episode


def test_physical_core_selection_does_not_count_smt_twice():
    topology = [(i, 0, i // 2, 5000) for i in range(16)]
    groups = physical_cpu_groups(set(range(16)), 2, 4, topology)
    assert len(groups) == 2
    cpus = [c for g in groups for c in g]
    assert len(cpus) == 4 and len({c // 2 for c in cpus}) == 4
    assert {0, 1, 2, 3}.isdisjoint(cpus)  # two physical cores reserved


def test_fifo_is_global_across_tasks_and_excludes_active(tmp_path):
    from arx_x5_data.recording import atomic_json

    paths = []
    for task, stamp in (("z", 1), ("a", 2), ("b", 3)):
        path = tmp_path / task / "episode_000001" / "job.json"
        path.parent.mkdir(parents=True)
        atomic_json(path, {"state": "queued", "enqueued_at_s": stamp})
        paths.append(path)
    assert queue_paths(tmp_path) == paths
    assert queue_paths(tmp_path, [paths[0]]) == paths[1:]


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ARX_TEST_LEROBOT") != "1", reason="Opt in")
def test_prevalidated_conversion_skips_second_decode_and_detects_mutation(
    make_episode, tmp_path, monkeypatch
):
    from arx_x5_data.export import convert

    source = make_episode((0, 1, 4, 5))
    verified = validate_episode(source, allow_mock=True)

    def no_second_validation(*args, **kwargs):
        raise AssertionError("Must reuse validation proof from the same worker")

    monkeypatch.setattr("arx_x5_data.export.discover_episodes", no_second_validation)
    result = convert(
        source, tmp_path / "good", "local/proof", allow_mock=True, _validated_episode=verified
    )
    assert result["metrics"]["validation_reused"] is True
    assert result["metrics"]["statistics_rgb_frames"] == 12
    with (source / "samples.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed after validation"):
        convert(
            source, tmp_path / "bad", "local/proof", allow_mock=True, _validated_episode=verified
        )
    assert not (tmp_path / "bad").exists()


def test_missing_frequency_info_still_deduplicates_smt(monkeypatch):
    import re
    from pathlib import Path

    def topology_file(path, *args, **kwargs):
        cpu = int(re.search(r"cpu(\d+)", str(path)).group(1))
        if path.name == "physical_package_id":
            return "0"
        if path.name == "core_id":
            return str(cpu // 2)
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "read_text", topology_file)
    assert physical_cpu_groups(set(range(8)), 2, 2) == [[4], [6]]
