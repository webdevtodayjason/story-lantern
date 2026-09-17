"""Offline harness: run the whole Story Lantern pipeline against a fake device.

    python3 tests/fake_device_test.py

No Tiiny required and nothing touches the network. The fake stands in for
DeviceWorker._http - the single socket - and asserts that no two device calls
ever overlap, which is the invariant the whole design rests on.

Exercises: the plan call, the empty-content/reasoning-scavenge path, 150004
backoff, per-page build, safety SOFTEN + one regeneration then fallback,
prefetch gating, every HTTP route, SSE, media serving, graceful degradation
when the illustrator is evicted or the voice is gone, and the character bible
staying frozen across two stories (including a fuzzy-matched misspelling).

The storyteller choice has its own file, tests/storyteller_test.py. What this one
holds it to is that a story told by a model chosen at run time still comes out
the same, on every route.
"""
import json, os, shutil, sys, tempfile, threading, time, urllib.error, urllib.request

HOME = tempfile.mkdtemp(prefix="lantern-test-")
# TIINY_BASE is the top of the resolver chain, so it beats anything the machine
# running the tests happens to have in ~/.tiinyapps/device.json. Without that,
# a developer with a real Tiiny planted by the farm would find this harness
# pointed at their actual hardware.
os.environ["TIINY_BASE"] = "http://127.0.0.1:8899"
os.environ["TIINY_KEY"] = "fake"
os.environ["LANTERN_HOME"] = HOME
os.environ["PORT"] = "8499"
os.environ.pop("TIINY_HOST", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lantern as L  # noqa: E402
from fake_models import FakeModels, CHAT, EMBED, TTS  # noqa: E402

# Pin the device outright: nothing in this harness may touch a network, and that
# includes the gateway port probe.
L.device.set_current(L.device.Device(
    host="127.0.0.1", port=8899, key="fake", source="the offline harness",
    plane="given", serial="FAKE-SERIAL-0001"))

PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
MP3 = (b"\xff\xfb\x90\x00" + b"\x00" * 64)

# One chat model and an embedder, which is what a real device looks like at
# bedtime once something else has the budget. The lantern has to find the chat
# model on its own; Ornith is not here and is not required.
MODELS = FakeModels(running=[CHAT, EMBED], npu_available=16)

calls = {"n": 0, "busy": 0, "concurrent": 0}
inflight = threading.Semaphore(1)


def fake_http(self, job):
    """Stand in for the one socket. Asserts serialisation, injects 150004."""
    if not inflight.acquire(blocking=False):
        calls["concurrent"] += 1
        raise AssertionError("TWO CONCURRENT DEVICE CALLS")
    try:
        calls["n"] += 1
        time.sleep(0.01)
        # Inject one 150004 on the 3rd call to prove backoff works.
        if calls["n"] == 3 and calls["busy"] == 0:
            calls["busy"] += 1
            raise L.DeviceBusy('{"code":150004}')
        if job.kind == "image":
            return PNG
        if job.kind == "speech":
            return MP3
        if job.kind == "control":
            # Routed by path, not waved through. A control call that answers
            # {"ok": true} to everything tells the lantern the device is holding
            # no models at all, which is a different test than the one intended.
            answer = MODELS.control(job.path)
            return {"ok": True} if answer is None else answer
        body = job.body
        text = json.dumps(body["messages"])
        if "spine" in text and "first_page" in text:
            plan = {
                "title": "Biscuit and the Dragon Who Kept the Lights On",
                "setting": "forest",
                "characters": [
                    {"name": "Biscuit", "kind": "pet",
                     "descriptor": "a small scruffy wheat-colored terrier with one folded left ear, a red collar, and short legs",
                     "personality": "brave and bouncy, always first to sniff a new thing"},
                    {"name": "Ember", "kind": "invented",
                     "descriptor": "a very large round green dragon with soft gold eyes and tiny wings",
                     "personality": "gentle and worried, hides behind his own tail"}],
                "spine": ["they meet", "the dark comes", "a nightlight"],
                "first_page": {"text": " ".join(["word"] * 80),
                               "scene": "a terrier nose to nose with a huge worried dragon"},
            }
            # Simulate Ornith: content EMPTY, answer only in reasoning_content.
            return {"choices": [{"message": {
                "content": "",
                "reasoning_content": "Let me think about this. {\"scratch\": 1}\n"
                                     "Now the answer:\n```json\n" + json.dumps(plan) + "\n```"}}]}
        if "suspicious" in text or "cautious adult" in text:
            page_text = body["messages"][1]["content"]
            if "shadow teeth" in page_text:
                return {"choices": [{"message": {"content": json.dumps(
                    {"verdict": "SOFTEN", "reason": "menacing imagery at bedtime"})}}]}
            return {"choices": [{"message": {"content": '{"verdict":"ALLOW","reason":"fine"}'}}]}
        # page writer
        soften = "safety check" in text
        prose = ("Biscuit trotted through the soft trees. " * 4) if not soften else \
                ("The little lamp was warm and the night was soft. " * 4)
        if "page 2" in text and not soften:
            prose = "Night came down with shadow teeth over the little forest path. " * 3
        return {"choices": [{"message": {"content": json.dumps(
            {"text": prose, "scene": "a small dog under tall trees",
             "characters": ["Biscuit", "Ember"]})}}]}
    finally:
        inflight.release()


L.DeviceWorker._http = fake_http

L.CFG.child_name = "Maya"
L.CFG.child_age = 5
L.CFG.busy_max_wait = 20
L.CFG.prefetch_stall_s = 3
L.db_init()
lan = L.Lantern(L.CFG)

# ---- serve, so the HTTP surface is tested too -----------------------------
L.Handler.lantern = lan
httpd = L.LanternServer(("127.0.0.1", 8499), L.Handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()


def get(path, raw=False):
    with urllib.request.urlopen("http://127.0.0.1:8499" + path, timeout=10) as r:
        d = r.read()
        return d if raw else json.loads(d)


def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:8499" + path,
                                 data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


# ---- SSE reader ------------------------------------------------------------
seen = []


def sse():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8499/events", timeout=60) as r:
            for line in r:
                s = line.decode().strip()
                if s.startswith("event:"):
                    seen.append(s.split(":", 1)[1].strip())
    except Exception:
        pass


threading.Thread(target=sse, daemon=True).start()
time.sleep(0.3)

fails = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  " + str(extra)) if not cond else ""))
    if not cond:
        fails.append(name)


print("\n== story 1 ==")
r = post("/api/story", {"request": "a story about a dragon scared of the dark and my dog Biscuit",
                        "pages": 3})
sid = r["story_id"]
s = lan.session
deadline = time.time() + 60
while s.thread.is_alive() and time.time() < deadline:
    time.sleep(0.1)

check("no concurrent device calls", calls["concurrent"] == 0)
check("150004 was hit and survived", calls["busy"] == 1)
check("story finished", s.status == "finished", s.status)
check("3 pages built", len(s.pages) == 3, len(s.pages))
check("title from reasoning_content scavenge", s.title.startswith("Biscuit and the Dragon"), s.title)
check("page 1 clamped to 60 words", len(s.pages[0].text.split()) == 60, len(s.pages[0].text.split()))
check("page 2 softened + regenerated", s.pages[1].verdict == "SOFTENED", s.pages[1].verdict)
check("regen counted", s.pages[1].regen == 1)
check("every page has audio", all(p.audio_path for p in s.pages.values()))
check("every page has image", all(p.image_path for p in s.pages.values()))
check("storyteller chosen from what was running", s.storyteller == CHAT, s.storyteller)
check("the voice model was started, not assumed", TTS in MODELS.starts, MODELS.starts)

doc = get(f"/api/story/{sid}")
check("api story pages", len(doc["pages"]) == 3, len(doc["pages"]))
check("story json names its storyteller", doc["storyteller"] == CHAT, doc["storyteller"])
check("a finished story has no failure reason", doc["reason"] is None, doc["reason"])
check("api safety events recorded", len(doc["safety_events"]) >= 1, doc["safety_events"])
check("image prompt carries frozen descriptor",
      "one folded left ear" in doc["pages"][0]["image_prompt"], doc["pages"][0]["image_prompt"][:120])
png = get(doc["pages"][0]["image_url"], raw=True)
mp3 = get(doc["pages"][0]["audio_url"], raw=True)
check("png served", png[:4] == b"\x89PNG")
check("mp3 served", mp3[:2] == b"\xff\xfb")

chars = get("/api/characters")["characters"]
check("2 characters in bible", len(chars) == 2, [c["name"] for c in chars])
biscuit = [c for c in chars if c["name"] == "Biscuit"][0]
seed1, desc1 = biscuit["art_seed"], biscuit["descriptor"]
check("character locked", biscuit["locked"])

st = get("/api/status")
check("status has worker", st["worker"]["calls"] > 0, st["worker"])
check("sse delivered events", "page.ready" in seen and "story.finished" in seen, seen[:12])

print("\n== story 2: does Biscuit come back identical? ==")


def fake_http2(self, job):
    """Second story: the model tries to invent a NEW look for Biscuit."""
    if job.kind in ("image", "speech", "control"):
        return fake_http(self, job)
    text = json.dumps(job.body["messages"])
    if "spine" in text and "first_page" in text:
        plan = {"title": "Biscuit Goes to the Moon", "setting": "moon",
                "characters": [{"name": "Biskit", "kind": "pet",
                                "descriptor": "a HUGE black poodle with a blue hat",
                                "personality": "sleepy"}],
                "spine": ["up", "down"],
                "first_page": {"text": "Biscuit floated gently past the quiet moon. " * 3,
                               "scene": "a dog floating past the moon"}}
        return {"choices": [{"message": {"content": json.dumps(plan)}}]}
    return fake_http(self, job)


L.DeviceWorker._http = fake_http2
r2 = post("/api/story", {"request": "biskit goes to the moon", "pages": 2})
s2 = lan.session
deadline = time.time() + 60
while s2.thread.is_alive() and time.time() < deadline:
    time.sleep(0.1)

chars2 = get("/api/characters")["characters"]
b2 = [c for c in chars2 if c["name"] == "Biscuit"]
check("no duplicate Biscuit row", len(chars2) == 2, [c["name"] for c in chars2])
check("fuzzy alias 'Biskit' resolved to Biscuit", bool(b2) and "Biskit" in b2[0]["aliases"],
      b2 and b2[0]["aliases"])
check("descriptor unchanged (frozen)", b2 and b2[0]["descriptor"] == desc1, b2 and b2[0]["descriptor"])
check("art_seed unchanged", b2 and b2[0]["art_seed"] == seed1)
check("appearances incremented", b2 and b2[0]["appearances"] >= 2, b2 and b2[0]["appearances"])
doc2 = get(f"/api/story/{r2['story_id']}")
check("story 2 reuses frozen descriptor in prompt",
      "one folded left ear" in doc2["pages"][0]["image_prompt"],
      doc2["pages"][0]["image_prompt"][:140])
check("same page seed for Biscuit across stories",
      s2._page_seed(s2.pages[0]) == seed1)

print("\n== degradation: image model gone ==")


def fake_http3(self, job):
    if job.kind == "image":
        raise L.DeviceError("HTTP 503 model evicted")
    return fake_http2(self, job)


L.DeviceWorker._http = fake_http3
r3 = post("/api/story", {"request": "a quiet snail story", "pages": 2})
s3 = lan.session
deadline = time.time() + 60
while s3.thread.is_alive() and time.time() < deadline:
    time.sleep(0.1)
check("story still finishes with no illustrator", s3.status == "finished", s3.status)
check("pages still narrated", all(p.audio_path for p in s3.pages.values()))
check("degraded event emitted", "page.degraded" in seen)

print("\n== degradation: the voice is gone ==")


def fake_http4(self, job):
    # The voice model evicted mid-story. The design says the words stay on
    # screen and a parent reads them; it does NOT say the story ends.
    if job.kind == "speech":
        raise L.DeviceError("HTTP 404 model_not_found")
    return fake_http2(self, job)


L.DeviceWorker._http = fake_http4
post("/api/story", {"request": "a story about a quiet moth", "pages": 2})
s5 = lan.session
deadline = time.time() + 60
while s5.thread.is_alive() and time.time() < deadline:
    time.sleep(0.1)
check("story still finishes with no voice", s5.status == "finished", s5.status)
check("pages still have words", all(p.text for p in s5.pages.values()))
check("pages still have plates", all(p.image_path for p in s5.pages.values()))
check("no page was narrated", not any(p.audio_path for p in s5.pages.values()))
check("silent pages still reached the lamp",
      len(get("/api/state")["pages"]) == len(s5.pages), get("/api/state"))

print("\n== stop mid-story ==")
L.DeviceWorker._http = fake_http2
r4 = post("/api/story", {"request": "a long slow story", "pages": 8})
time.sleep(0.5)
post("/api/stop", {})
time.sleep(1.0)
row = get("/api/stories")["stories"][0]
check("stopped story recorded", row["status"] in ("stopped", "finished"), row["status"])

print("\n== lamp UI contract (static/show.html) ==")
check("hello sent on connect", "hello" in seen, seen[:5])
for name in ("state", "story", "page", "end"):
    check(f"UI event '{name}' emitted", name in seen, seen[:20])
st2 = get("/api/state")
check("/api/state shape",
      set(st2) == {"state", "story", "pages", "storyteller", "reason"}, list(st2))
check("idle /api/state carries no failure", st2["reason"] is None, st2["reason"])
req = urllib.request.Request("http://127.0.0.1:8499/api/request",
                             data=json.dumps({"text": "a story about a slow snail"}).encode(),
                             method="POST", headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10) as r:
    check("/api/request returns 202", r.status == 202, r.status)
    body202 = json.loads(r.read())
check("/api/request returns ok:true", body202.get("ok") is True, body202)
lan.session.stop()
time.sleep(0.3)

print("\n== replay a saved story (the shelf) ==")
shelf = get("/api/stories")["stories"]
mine = [x for x in shelf if x["id"] == sid][0]
check("shelf lists the story", mine["pages"] > 0, mine["pages"])
check("shelf entry carries a cover", bool(mine["cover"]), mine["cover"])

rp = post(f"/api/story/{sid}/replay", {})
check("replay accepted", rp.get("ok") is True, rp)
check("replay reports every page", rp.get("pages") == mine["pages"], rp)
st = get("/api/state")
check("replay puts the lamp back in telling", st["state"] == "telling", st["state"])
check("replay re-serves the pages", len(st["pages"]) == mine["pages"], len(st["pages"]))
check("replay page 0 keeps its art and voice",
      bool(st["pages"][0]["image_url"]) and bool(st["pages"][0]["audio_url"]),
      st["pages"][0])
check("last page is marked last", st["pages"][-1].get("last") is True, st["pages"][-1])
# A replay must never call the device: the story already exists.
calls_before = L.db().execute("SELECT COUNT(*) c FROM device_call").fetchone()["c"]
post(f"/api/story/{sid}/replay", {})
check("replay makes no device calls",
      L.db().execute("SELECT COUNT(*) c FROM device_call").fetchone()["c"] == calls_before)
post("/api/stop", {})
check("stop returns to the candle", get("/api/state")["state"] == "idle")

# The bug this catches: ReplaySession was missing StorySession's `thread`, so
# every new story request after a replay died with AttributeError and the lamp
# sat on a candle forever.
post(f"/api/story/{sid}/replay", {})
again = post("/api/request", {"request": "a story about a kite"})
check("a new story can start right after a replay", again.get("ok") is True, again)
post("/api/stop", {})

print("\n== parent page contract (static/parent.html) ==")
pd = get("/api/parent/data")
for k in ("child", "stories"):
    check(f"/api/parent/data has {k}", k in pd, list(pd))
check("parent data carries stories with pages",
      bool(pd["stories"]) and "pages" in pd["stories"][0], list(pd["stories"][:1]))
told = [s for s in pd["stories"] if s.get("storyteller")]
check("parent log says who told the story", bool(told),
      [(s["id"], s.get("storyteller")) for s in pd["stories"][:3]])
verbatim = [e for s in pd["stories"] for e in s.get("safety_events", [])
            if e.get("offending_text")] if "stories" in pd else []
check("parent log keeps offending text verbatim", bool(verbatim), verbatim[:1])

n_chars_before = len(get("/api/characters")["characters"])
dele = post(f"/api/parent/story/{sid}/delete", {})
check("story deleted", dele.get("ok") is not False, dele)
check("bible survives a story deletion",
      len(get("/api/characters")["characters"]) == n_chars_before)
try:
    get(f"/api/story/{sid}")
    check("deleted story is gone", False)
except urllib.error.HTTPError as e:
    check("deleted story is gone", e.code == 404, e.code)

print("\n== media route ==")
try:
    get("/media/pageturn.wav", raw=True)
    check("optional ambience 404s harmlessly", False, "should not exist")
except urllib.error.HTTPError as e:
    check("optional ambience 404s harmlessly", e.code == 404, e.code)
try:
    get("/media/../../etc/passwd", raw=True)
    check("path traversal refused", False)
except urllib.error.HTTPError as e:
    check("path traversal refused", e.code == 404, e.code)

print("\n== transcript log ==")
n = L.db().execute("SELECT COUNT(*) c, SUM(busy_waits) b FROM device_call").fetchone()
check("device_call rows written", n["c"] > 10, n["c"])
check("busy waits logged", (n["b"] or 0) >= 1, n["b"])

httpd.shutdown()
lan.shutdown()
shutil.rmtree(HOME, ignore_errors=True)
print("\nFAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
