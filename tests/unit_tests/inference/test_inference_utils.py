from types import SimpleNamespace

import torch

from megatron.core.inference import utils as inference_utils
from megatron.core.inference.utils import Counter


class TestInferenceUtils:

    def test_counter(self):
        counter = Counter()
        r = next(counter)
        assert r == 0, f'Counter return value should be 0 but it is {r}'
        assert counter.counter == 1, f'Counter should be 1 but it is {counter.counter}'
        counter.reset()
        assert counter.counter == 0, f'Counter should be 0 but it is {counter.counter}'

    def test_moe_metadata_sync_has_decoder_and_mtp_leaders(self, monkeypatch):
        """Decoder and serial MTP must independently refresh shared MoE metadata."""

        class FakeMoELayer(torch.nn.Module):

            def __init__(self, is_mtp_layer):
                super().__init__()
                self.is_mtp_layer = is_mtp_layer
                self._inference_token_dispatcher = SimpleNamespace(_runs_metadata_sync=None)

        class FakeModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [
                        FakeMoELayer(False),
                        FakeMoELayer(False),
                        FakeMoELayer(True),
                        FakeMoELayer(True),
                    ]
                )

        monkeypatch.setattr("megatron.core.transformer.moe.moe_layer.MoELayer", FakeMoELayer)
        model = FakeModel()

        inference_utils.set_moe_metadata_sync(model)

        assert [
            layer._inference_token_dispatcher._runs_metadata_sync for layer in model.layers
        ] == [True, False, True, False]

    def test_moe_metadata_sync_cache_is_scoped_to_model(self, monkeypatch):
        """Building a second inference model must configure its dispatchers too."""

        class FakeMoELayer(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.is_mtp_layer = False
                self._inference_token_dispatcher = SimpleNamespace(_runs_metadata_sync=None)

        monkeypatch.setattr("megatron.core.transformer.moe.moe_layer.MoELayer", FakeMoELayer)
        first = torch.nn.Sequential(FakeMoELayer())
        second = torch.nn.Sequential(FakeMoELayer())

        inference_utils.set_moe_metadata_sync(first)
        inference_utils.set_moe_metadata_sync(second)

        assert first[0]._inference_token_dispatcher._runs_metadata_sync is True
        assert second[0]._inference_token_dispatcher._runs_metadata_sync is True
