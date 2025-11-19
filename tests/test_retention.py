import pytest
from hf_lifecycle.retention import (
    KeepLastN,
    KeepBestM,
    CombinedRetentionPolicy,
    CustomRetentionPolicy,
)


class TestKeepLastN:
    def test_keep_last_n_basic(self):
        policy = KeepLastN(3)
        checkpoints = [
            {"name": "ckpt-1", "step": 100},
            {"name": "ckpt-2", "step": 200},
            {"name": "ckpt-3", "step": 300},
            {"name": "ckpt-4", "step": 400},
            {"name": "ckpt-5", "step": 500},
        ]
        
        to_keep = policy.select_checkpoints_to_keep(checkpoints)
        
        assert len(to_keep) == 3
        assert "ckpt-5" in to_keep
        assert "ckpt-4" in to_keep
        assert "ckpt-3" in to_keep

    def test_keep_last_n_fewer_checkpoints(self):
        policy = KeepLastN(5)
        checkpoints = [
            {"name": "ckpt-1", "step": 100},
            {"name": "ckpt-2", "step": 200},
        ]
        
        to_keep = policy.select_checkpoints_to_keep(checkpoints)
        
        assert len(to_keep) == 2

    def test_keep_last_n_empty(self):
        policy = KeepLastN(3)
        to_keep = policy.select_checkpoints_to_keep([])
        assert len(to_keep) == 0

    def test_keep_last_n_invalid(self):
        with pytest.raises(ValueError):
            KeepLastN(0)


class TestKeepBestM:
    def test_keep_best_m_min_mode(self):
        policy = KeepBestM(2, metric="loss", mode="min")
        checkpoints = [
            {"name": "ckpt-1", "step": 100, "metrics": {"loss": 0.5}},
            {"name": "ckpt-2", "step": 200, "metrics": {"loss": 0.3}},
            {"name": "ckpt-3", "step": 300, "metrics": {"loss": 0.4}},
            {"name": "ckpt-4", "step": 400, "metrics": {"loss": 0.6}},
        ]
        
        to_keep = policy.select_checkpoints_to_keep(checkpoints)
        
        assert len(to_keep) == 2
        assert "ckpt-2" in to_keep  # loss=0.3
        assert "ckpt-3" in to_keep  # loss=0.4

    def test_keep_best_m_max_mode(self):
        policy = KeepBestM(2, metric="accuracy", mode="max")
        checkpoints = [
            {"name": "ckpt-1", "step": 100, "metrics": {"accuracy": 0.8}},
            {"name": "ckpt-2", "step": 200, "metrics": {"accuracy": 0.95}},
            {"name": "ckpt-3", "step": 300, "metrics": {"accuracy": 0.9}},
            {"name": "ckpt-4", "step": 400, "metrics": {"accuracy": 0.85}},
        ]
        
        to_keep = policy.select_checkpoints_to_keep(checkpoints)
        
        assert len(to_keep) == 2
        assert "ckpt-2" in to_keep  # accuracy=0.95
        assert "ckpt-3" in to_keep  # accuracy=0.9

    def test_keep_best_m_missing_metric(self):
        policy = KeepBestM(2, metric="loss", mode="min")
        checkpoints = [
            {"name": "ckpt-1", "step": 100, "metrics": {"accuracy": 0.8}},
            {"name": "ckpt-2", "step": 200, "metrics": {}},
        ]
        
        to_keep = policy.select_checkpoints_to_keep(checkpoints)
        
        assert len(to_keep) == 0

    def test_keep_best_m_invalid_mode(self):
        with pytest.raises(ValueError):
            KeepBestM(2, metric="loss", mode="invalid")

    def test_keep_best_m_invalid_m(self):
        with pytest.raises(ValueError):
            KeepBestM(0, metric="loss")


class TestCombinedRetentionPolicy:
    def test_combined_policy(self):
        # Keep last 2 OR best 2 by loss
        policy1 = KeepLastN(2)
        policy2 = KeepBestM(2, metric="loss", mode="min")
        combined = CombinedRetentionPolicy([policy1, policy2])
        
        checkpoints = [
            {"name": "ckpt-1", "step": 100, "metrics": {"loss": 0.9}},
            {"name": "ckpt-2", "step": 200, "metrics": {"loss": 0.3}},  # best
            {"name": "ckpt-3", "step": 300, "metrics": {"loss": 0.4}},  # 2nd best & last 2
            {"name": "ckpt-4", "step": 400, "metrics": {"loss": 0.8}},  # last 2
        ]
        
        to_keep = combined.select_checkpoints_to_keep(checkpoints)
        
        # Should keep union: ckpt-2 (best), ckpt-3 (2nd best & last 2), ckpt-4 (last 2)
        assert len(to_keep) == 3
        assert "ckpt-2" in to_keep
        assert "ckpt-3" in to_keep
        assert "ckpt-4" in to_keep


class TestCustomRetentionPolicy:
    def test_custom_policy(self):
        # Custom policy: keep checkpoints with step divisible by 100
        def divisible_by_100(checkpoints):
            return [
                ckpt["name"] 
                for ckpt in checkpoints 
                if ckpt.get("step", 0) % 100 == 0
            ]
        
        policy = CustomRetentionPolicy(divisible_by_100)
        checkpoints = [
            {"name": "ckpt-1", "step": 100},
            {"name": "ckpt-2", "step": 150},
            {"name": "ckpt-3", "step": 200},
            {"name": "ckpt-4", "step": 250},
        ]
        
        to_keep = policy.select_checkpoints_to_keep(checkpoints)
        
        assert len(to_keep) == 2
        assert "ckpt-1" in to_keep  # step=100
        assert "ckpt-3" in to_keep  # step=200
