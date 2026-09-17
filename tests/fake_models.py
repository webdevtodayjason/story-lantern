"""The model-lifecycle half of a fake Tiiny: what it holds, and what it will load.

Shared by fake_device_test.py and storyteller_test.py. Both need the same four
answers - /api/v1/models/running, /v1/models, /api/v1/models/ and the NPU budget -
and a second copy of those shapes is a second place to get them wrong.

Two behaviours here are not decoration and the tests depend on them:

* A start whose npu_usage does not fit is accepted with a 200 and then ROLLED
  BACK - the model never appears in the running list. That is what the real
  device does, silently, and it is why the lantern checks the budget before it
  ever posts a start.
* One catalog row deliberately has no `supports_chat` key, so the picker's
  fallback to the model `type` is exercised rather than assumed.
"""
import urllib.parse

ORNITH = "deepreinforce-ai/Ornith-1.0-35B"
BIG_CHAT = "Qwen/Qwen3.8-27B"
CHAT = "Qwen/Qwen3-8B"
CODER = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
IMAGE = "Tongyi-MAI/Z-Image-Turbo"
EMBED = "Qwen/Qwen3-Embedding-0.6B"
RERANK = "Qwen/Qwen3-Reranker-0.6B"
OCR = "PaddlePaddle/PP-DocLayoutV3"
ASR = "Qwen/Qwen3-ASR-1.7B"

# id -> (type, supports_chat or None, npu_usage). None means the firmware did not
# say, which is the case the type fallback exists for.
SPECS = {
    ORNITH: ("Text Generation", True, 50),
    BIG_CHAT: ("Image-Text-to-Text", None, 55),
    CHAT: ("Text Generation", True, 12),
    CODER: ("Text Generation", True, 30),
    TTS: ("Text-to-Speech", False, 7),
    IMAGE: ("Text-to-Image", False, 32),
    EMBED: ("Text Embedding", False, 1),
    RERANK: ("Text Reranking", False, 2),
    OCR: ("Image-to-Text", False, 4),
    ASR: ("ASR", False, 7),
}

CAPABILITY = {"Text Generation": "main", "Image-Text-to-Text": "main",
              "Text-to-Speech": "voice", "Text-to-Image": "image",
              "Text Embedding": "embedding", "Text Reranking": "rerank",
              "Image-to-Text": "ocr", "ASR": "audio"}


class FakeModels:
    """The device's model state, mutated by start and stop like the real one."""

    def __init__(self, running=(), installed=None, npu_available=16):
        self.running = list(running)
        self.installed = list(installed if installed is not None else SPECS)
        self.npu_available = npu_available
        self.starts = []
        self.stops = []

    # ---- what the four read routes answer ---------------------------------

    def _catalog_row(self, mid):
        kind, chat, _ = SPECS[mid]
        row = {"id": mid, "type": kind, "supports_vision": kind.startswith("Image"),
               "capabilities": [CAPABILITY[kind]], "supported": []}
        if chat is not None:
            row["supports_chat"] = chat
        return row

    def _installed_row(self, mid):
        kind, _, npu = SPECS[mid]
        return {"fullname": mid, "name": mid.split("/")[-1], "type": kind,
                "npu_usage": npu, "status": "downloaded"}

    def control(self, path):
        """Answer a model route, or return None if this is not one."""
        if path == "/api/v1/models/running":
            return {"running": list(self.running),
                    "instances": [{"id": m, "port": 9098} for m in self.running],
                    "pending": []}
        if path == "/v1/models":
            return {"data": [self._catalog_row(m) for m in SPECS]}
        if path == "/api/v1/models/":
            return {"data": [self._installed_row(m) for m in self.installed]}
        if path.endswith("/npu/status"):
            return {"npu_total": 100, "npu_used": 100 - self.npu_available,
                    "npu_available": self.npu_available}
        if path.startswith("/api/v1/models/") and path.endswith("/start"):
            return self._start(self._id_in(path, "/start"))
        if path.startswith("/api/v1/models/") and path.endswith("/stop"):
            return self._stop(self._id_in(path, "/stop"))
        return None

    # ---- lifecycle ---------------------------------------------------------

    @staticmethod
    def _id_in(path, suffix):
        enc = path[len("/api/v1/models/"):-len(suffix)]
        return urllib.parse.unquote(enc)

    def _start(self, mid):
        self.starts.append(mid)
        if mid in self.running:
            return {"message": f"{mid} already running"}
        npu = SPECS.get(mid, (None, None, 100))[2]
        if mid in self.installed and npu <= self.npu_available:
            self.running.append(mid)
            self.npu_available -= npu
        # No error either way. A load that does not fit is rolled back where
        # nobody can see it, which is the whole reason the lantern asks first.
        return {"message": f"start loading {mid}", "progress": 0}

    def _stop(self, mid):
        self.stops.append(mid)
        if mid in self.running:
            self.running.remove(mid)
            self.npu_available += SPECS.get(mid, (None, None, 0))[2]
        return {"removed_container_ids": []}
