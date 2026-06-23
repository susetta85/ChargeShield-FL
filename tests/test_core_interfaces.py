# tests/test_core_interfaces.py
import pytest
from src.core.base_node import AbstractChargingNode, NodeConfig
from src.core.base_adapter import AbstractProtocolAdapter
from src.core.base_dataset import AbstractDataset
from src.core.base_auditor import AbstractPrivacyAuditor, AuditReport


# --- Concrete stubs (solo per i test) ---

class DummyNode(AbstractChargingNode):
    def collect_data(self): return {"voltage": 230.0, "current": 16.0}
    def preprocess(self, raw): return {k: float(v) for k, v in raw.items()}
    def get_status(self): return "online"

class DummyAdapter(AbstractProtocolAdapter):
    def encode(self, data): return str(data).encode()
    def decode(self, raw): return {"decoded": True}
    def get_protocol_name(self): return "DUMMY_v1"

class DummyDataset(AbstractDataset):
    def load(self, path): self._data = [{"kw": 7.4}]
    def get_sample(self, index): return self._data[index]
    def __len__(self): return 1
    def get_feature_names(self): return ["kw"]

class DummyAuditor(AbstractPrivacyAuditor):
    def audit(self, node_id, round_id, model_update):
        return AuditReport(node_id=node_id, round_id=round_id, privacy_score=0.95, epsilon=0.1)
    def reset(self): pass
    def get_cumulative_epsilon(self, node_id): return 0.1


# --- Test AbstractChargingNode ---

def test_node_collect_data():
    node = DummyNode(NodeConfig(node_id="highway-01", cluster_id="highway", location="A1"))
    data = node.collect_data()
    assert "voltage" in data
    assert "current" in data

def test_node_preprocess():
    node = DummyNode(NodeConfig(node_id="highway-01", cluster_id="highway", location="A1"))
    result = node.preprocess({"voltage": "230.0"})
    assert isinstance(result["voltage"], float)

def test_node_status():
    node = DummyNode(NodeConfig(node_id="urban-01", cluster_id="urban", location="B2"))
    assert node.get_status() == "online"

def test_node_config():
    config = NodeConfig(node_id="corporate-01", cluster_id="corporate", location="C3")
    node = DummyNode(config)
    assert node.config.cluster_id == "corporate"


# --- Test AbstractProtocolAdapter ---

def test_adapter_encode_decode():
    adapter = DummyAdapter()
    data = {"voltage": 230.0}
    encoded = adapter.encode(data)
    assert isinstance(encoded, bytes)
    decoded = adapter.decode(encoded)
    assert isinstance(decoded, dict)

def test_adapter_protocol_name():
    adapter = DummyAdapter()
    assert adapter.get_protocol_name() == "DUMMY_v1"


# --- Test AbstractDataset ---

def test_dataset_load_and_len():
    ds = DummyDataset()
    ds.load("fake/path")
    assert len(ds) == 1

def test_dataset_get_sample():
    ds = DummyDataset()
    ds.load("fake/path")
    sample = ds.get_sample(0)
    assert "kw" in sample

def test_dataset_feature_names():
    ds = DummyDataset()
    ds.load("fake/path")
    assert "kw" in ds.get_feature_names()


# --- Test AbstractPrivacyAuditor ---

def test_auditor_returns_report():
    auditor = DummyAuditor()
    report = auditor.audit("highway-01", 1, {"weights": [0.1, 0.2]})
    assert isinstance(report, AuditReport)
    assert report.privacy_score == 0.95
    assert report.epsilon == 0.1

def test_auditor_cumulative_epsilon():
    auditor = DummyAuditor()
    assert auditor.get_cumulative_epsilon("highway-01") == 0.1
