"""Smoke tests for the assistant tools, run against the compact demo data in artifacts/
(the same data the deployed apps use). Values match the evaluation's expected answers."""
import pytest

from src import tools as T


def test_list_drive_cycles():
    r = T.list_drive_cycles()
    assert r["n_cycles"] == 69
    assert round(r["total_hours"]) == 185


def test_cycle_summary_known_test_cycle():
    s = T.get_cycle_summary(74)
    assert s["split"] == "test"
    assert s["components"]["magnet"]["peak_measured_c"] == pytest.approx(91.9, abs=0.1)


def test_unknown_cycle_raises_helpful_error():
    with pytest.raises(ValueError, match="does not exist"):
        T.get_cycle_summary(999)


def test_hot_restarts():
    ids = {c["profile_id"] for c in T.find_hot_restarts(10)["cycles"]}
    assert ids == {53, 47, 48}


def test_overheating_measured_vs_sensor():
    measured = {c["profile_id"] for c in T.find_overheating(100, "magnet", "measured")["cycles"]}
    estimated = {c["profile_id"] for c in T.find_overheating(100, "magnet", "virtual_sensor")["cycles"]}
    assert measured == {20, 24, 27, 31}
    assert measured - estimated == {31}  # the documented miss


def test_what_if_runs_the_model():
    r = T.what_if(62, coolant_delta_c=10, current_scale=1.0)
    assert r["magnet"]["peak_change_k"] == pytest.approx(4.9, abs=0.1)
    assert any("not measurements" in w for w in r["warnings"])  # every what-if says it is an estimate


def test_what_if_flags_extrapolation():
    r = T.what_if(74, coolant_delta_c=0, current_scale=1.3)  # currents beyond anything in the data
    assert any("extrapolation" in w for w in r["warnings"])


@pytest.mark.parametrize("kwargs", [{"coolant_delta_c": 50}, {"current_scale": 3.0}])
def test_what_if_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        T.what_if(62, **{"coolant_delta_c": 0, "current_scale": 1.0, **kwargs})
