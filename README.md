# Story Lantern

A bedside lamp your kid talks to.

They ask for a story. The lantern shows its work while it writes one, and half a minute
later a painted page appears in an open book and a warm voice starts reading. Their
characters come back next time looking and acting the same. The dog Biscuit still has the
folded ear and the red collar.

Everything happens on a **Tiiny Pocket** on your own network. The story is written,
illustrated and narrated by a box on the nightstand. No account, no subscription, no cloud
API, and nothing about your child leaves the house.

---

## What it actually does

| | |
|---|---|
| **Writes** the story | Ornith-1.0-35B, page by page, ~60 words a page |
| **Paints** every page | Z-Image-Turbo, 512×512, 8 diffusion steps, ~8s |
| **Reads** it aloud | Qwen3-TTS CustomVoice - a warm, unhurried voice |
| **Remembers** characters | a character bible: a fixed descriptive phrase plus a fixed image seed, replayed verbatim into every later illustration |
| **Checks** every page | a safety layer that classifies each page *before* it can be shown or spoken |
| **Shows parents** everything | a transparency log: what was asked, what was written, and the verdict on every page |

The character memory is the product. A five-year-old notices when the dog looks right.

---

## What it looks like

**It is a book.** Two pages with a spine between them: the illustration on the left, the
story on the right, set with the wider margin at the gutter the way a bound book actually
is. When the page turns, a sheet pivots on the spine and **its back carries the next
illustration**, because that is how a real leaf works. The front you were reading rotates
away, the picture you are turning to arrives on the back of it, and the next page's words
are waiting underneath. It is not a cross-fade with a rotation bolted on.

**It shows its work while it writes.** The first page takes about half a minute, and a
candle sitting there doing nothing for half a minute reads as broken. So the lantern says
what it is doing, out of real events, never a simulated progress bar: it dreams up the
story, then the whole storyboard lays itself out one plate per page with the actual beats
written on them, then the plate being made right now says whether it is being written,
checked, voiced or painted.

**Three buttons a five-year-old can hit without aiming.** Pause and play, a star for a new
story, and a book for every story you have already told. Pause holds both halves of a page:
the voice and the clock that turns the page when there is no voice.

**A shelf of saved stories.** Every story is kept with its pictures and its narration. Tap
one and it replays straight from the database, instantly, with no device calls at all. A
child asking for the dragon story again should not wait half a minute for a story that
already exists.

**A child never sees an error.** If an illustration fails, the left page becomes a printed
blank with a keyline and an ornament, the way a book with a missing plate looks. If
narration fails, the words stay on screen long enough to be read aloud. No spinners, no
percentages, no stack traces anywhere in the display.

---

## Requirements

- A **Tiiny Pocket** reachable on your network
- Any machine to host the lantern: a Raspberry Pi, a mini PC, an old laptop
- **Python 3.9 or newer** - standard library only, no pip install, no virtualenv, no build
  step (verified on 3.9.6 and 3.13)
- A browser for the display. Kiosk mode on a small landscape screen makes it a lamp

## The three models

Load these two in TiinyOS **before** you start the lantern. Nothing else needs to be
running, and the lantern will not load them for you:

| Model | Does | NPU |
|---|---|---|
| `deepreinforce-ai/Ornith-1.0-35B` | writes the story, and checks it | 50u |
| `Tongyi-MAI/Z-Image-Turbo` | paints each page | 32u |

The third, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (7u), is the voice. The lantern starts it
when a story begins and releases it after twenty minutes idle, so it is not holding budget
between bedtimes. Set `manage_tts: false` in `config.json` if you would rather load it
yourself and leave it up.

That is 89 of the device's 100 units with the voice in. If something else is holding
budget, unload it first or the image model will refuse to start. Check what is up with
`curl localhost:8420/api/models` once the lantern is running.

---

## Install

```bash
git clone https://github.com/webdevtodayjason/story-lantern
cd story-lantern

export TIINY_HOST=192.168.1.50          # your device, host only, no scheme and no port
export TIINY_KEY=<your device api key>  # Settings → API in TiinyOS

python3 lantern.py --selfcheck          # one short story end to end, prints timings
python3 lantern.py                      # serve it
```

`--selfcheck` is the install test. It writes, checks, voices and paints a real story and
prints what each step cost, so a bad key or a model that is not loaded fails in a few
seconds with a clear reason instead of at bedtime.

Then open **http://localhost:8420/show** for the lantern and
**http://localhost:8420/parent** for the transparency log.

### Look at it with no device

```
http://localhost:8420/show?demo=1&go
```

A fake story with canvas-painted plates, no device and no network. It runs the same code
the real thing does, so it is the way to review the book, the page turn and the workshop
before you own a Tiiny, or to work on the display without burning device time. The lantern
still needs `TIINY_HOST` and `TIINY_KEY` set to start, but demo mode never calls the device.

### As an actual bedside lamp

```bash
chromium-browser --kiosk --autoplay-policy=no-user-gesture-required \
  --app=http://127.0.0.1:8420/show
```

**The autoplay flag is not optional.** Without it the narration silently never plays: the
browser blocks audio that no one clicked for, and there is nothing to click on a lamp.

Point it at a landscape screen. The book is a spread, so a tall phone-shaped display works
but gets small.

---

## Configuration

Environment, all optional except the two device settings:

| Variable | Default | Meaning |
|---|---|---|
| `TIINY_HOST` | - | device address (**required**), host only - `:8800` is added for you |
| `TIINY_KEY` | - | device API key (**required**) |
| `PORT` | `8420` | web port |
| `LANTERN_HOME` | `~/.lantern` | everything it keeps: database, generated media |
| `LANTERN_DB` | `$LANTERN_HOME/lantern.db` | SQLite state |
| `LANTERN_BLOCKLIST` | `./blocklist_extra.txt` | extra words for the deterministic backstop. Copy `blocklist_extra.example.txt` to start one; it is gitignored so yours is yours |
| `LANTERN_SAFETY_JSONL` | `$LANTERN_HOME/safety-events.jsonl` | append-only safety log |
| `LANTERN_TTS_IDLE_S` | `1200` | seconds before an idle voice model is released |
| `LANTERN_ALLOW_UNVERIFIED` | *unset* | **leave it unset.** See the safety section |
| `LANTERN_DEBUG` | *unset* | verbose device logging |

Copy `config.example.json` to `config.json`, beside the script or in `$LANTERN_HOME`, for
the rest: the child's name and age, how many pages a story runs to, whether the lantern
manages the TTS model, and whether the independent classifier runs. Environment wins over
the file.

---

## Troubleshooting

**`Set TIINY_HOST and TIINY_KEY`** - it will not start without both. `TIINY_HOST` is a bare
host or IP. Do not include `http://` or a port; the code appends `:8800`.

**The narration never plays.** The kiosk is missing
`--autoplay-policy=no-user-gesture-required`. In an ordinary browser tab, any click on the
page unlocks audio.

**The image model refuses to start.** Something else on the device is holding NPU budget.
`curl localhost:8420/api/models` lists what is running; unload whatever you do not need.

**Everything fails with `150004`.** That is the device saying it is already doing an
inference. The lantern retries with backoff and rides it out, but if another program is
hammering the same Tiiny you both lose. See the note on OneLane below.

**It hangs on "the lantern is dreaming".** The plan call can legitimately take half a
minute. After that there is a giveaway clock: if no page ever arrives the lantern ends
warmly rather than leaving a child staring at a candle.

**On macOS, `ERR_CONNECTION_REFUSED` from the browser and `EHOSTUNREACH` to the device.**
macOS tracks Local Network permission per binary. Homebrew's Python and the system Python
are two different binaries with two different grants. Either grant the one you are using in
System Settings → Privacy & Security → Local Network, or run it with `/usr/bin/python3`.

---

## The HTTP API

Everything the display uses is public and unauthenticated on the loopback port. Useful if
you want to drive it from a button, a cron job or your own front end.

| | |
|---|---|
| `GET /show`, `GET /parent` | the two views |
| `GET /events` | server-sent events: `story`, `page`, `build`, `state`, `notice`, `end` |
| `GET /api/state` | everything a browser needs to resume exactly where it was |
| `POST /api/request` `{"text": "..."}` | ask for a story |
| `POST /api/stop` | end the current story and return to the candle |
| `GET /api/stories` | the shelf: every story with its page count and cover |
| `GET /api/story/{id}` | one story, with pages and safety verdicts |
| `POST /api/story/{id}/replay` | play a saved story again, with no device calls |
| `POST /api/progress` | the display telling the engine which page is being read |
| `GET /api/models` | what is loaded on the device right now |
| `GET /api/parent/data` | the transparency log |
| `GET /healthz` | liveness |

---

## About the safety layer

An unmoderated model generating bedtime content for a five-year-old is the failure mode
that matters, so this is the part that got the most attention.

Three things happen before a child hears a word:

1. **A deterministic backstop** runs on the request before any model is involved. A
   catastrophic phrase never depends on a model being in a good mood.
2. **A charter check** on what the child asked for. If it needs declining, the lantern
   redirects warmly - it never shows a child an error or a refusal.
3. **A classifier pass on every generated page**, against a rubric written for ages 3–8:
   no violence beyond fairy-tale peril, no death of a named companion, no body horror, no
   adult themes, nothing that frightens a child at bedtime, nothing instructing real-world
   danger. A failed page is regenerated once with a gentler steer; if it fails again, a
   pre-written safe page is substituted and the substitution is logged loudly.

**It fails closed.** If the device cannot be reached to classify a page, the page is *not*
shown. `LANTERN_ALLOW_UNVERIFIED=1` takes the other side of that trade knowingly; the
default does not. This was a deliberate reversal during review - the earlier behaviour
showed unverified pages and merely badged them, which is the wrong default when the
audience is asleep in ten minutes.

Every verdict is persisted. The parent log hides nothing, including the pages that were
blocked and why.

This layer is not a guarantee. It is a serious, auditable effort, and you should read
`safety.py` yourself before pointing this at your own child. `python3 safety.py` runs an
adversarial self-test against a table of both innocent and genuinely nasty requests.

---

## What it costs, measured

One 8-page story on a Tiiny Pocket, from that story's own transcript log. Medians across
the eight pages:

| Step | Time |
|---|---|
| Plan the whole story (one call) | 23.3s |
| Write a page | 4.7s |
| Safety-check a page | 2.4s |
| Narrate a page | 8.7s |
| Paint a page | 7.9s |
| A whole page, start to finish | ~23.5s |
| **Ask → first page reading aloud** | **33.3s** |
| Device busy over the whole story | 210.9s across 32 calls |

**Only the first number is a wait.** A page takes about 23 seconds to make and about 27
seconds to read aloud, so from page two on the device is always a page ahead and there is
nothing to wait for. That is also why the total is not a useful number: generation is paced
by how fast the story is being read, not the other way round.

---

## Design notes worth knowing

**The device does one inference at a time.** A second concurrent call fails with error
`150004`. Every device call in this project goes through a single `DeviceWorker` thread and
retries `150004` with backoff, so Story Lantern co-exists with other applications using the
same Tiiny rather than fighting them.

If you are running several things against one Tiiny,
[OneLane](https://github.com/webdevtodayjason/onelane) does this properly across separate
programs rather than just within one. Retrying with backoff works, but it is still two apps
guessing about each other. OneLane lets them actually take turns.

**Prefetch is what makes it feel instant.** While page one's narration plays (~30 seconds),
the device is already writing and painting page two (~23 seconds). After the first page
there is no waiting, ever.

**A page ships as soon as it has a voice, and gets its picture about eight seconds later.**
The voice is what starts a page, so waiting for the illustration would cost eight seconds of
perceived latency on every page. The picture arrives afterwards and the page it belongs to
is repainted underneath it. That last half is easy to forget: an early version preloaded the
late illustration and never put it on screen, so almost every page kept the no-picture
treatment for its whole turn and the lamp looked like it never made pictures at all.

**512×512 is not an aesthetic choice.** On current firmware it is the only image size
Z-Image-Turbo will render; every other size fails after ~30 seconds.

**The chat model puts its reasoning in `reasoning_content`, and it counts against
`max_tokens`.** Ask for JSON with a small budget and you get an empty response. This project
uses a generous budget and scavenges the last complete JSON object out of the reasoning text
if `content` comes back empty.

**A replay is not a regeneration.** Saved stories are re-broadcast from SQLite through the
same event path a live story uses, so the display cannot tell the difference and the device
is never touched.

---

## Making it yours

The obvious extensions, roughly in order of how much fun they are:

- **Voice input.** ASR (`Qwen3-ASR-1.7B`, 7u) and a `getUserMedia` capture in the display,
  so the child speaks instead of typing. Left out of this version only because the NPU
  budget was shared with another application during development.
- **Your own art style.** The illustration prompt prefix is one constant. Swap gouache for
  crayon, woodcut, cut-paper.
- **A physical button.** A cheap arcade button that enumerates as a keyboard sending `Space`
  needs no GPIO code at all.
- **More voices.** A different narrator per character.
- **Other languages.** The story prompt and the safety rubric are the only places language
  is assumed.

---

## Tests

```bash
python3 tests/fake_device_test.py     # the whole engine against a fake device, no hardware
python3 safety.py                     # adversarial self-test on the safety layer
python3 lantern.py --selfcheck        # one real story, end to end, with timings
```

The first one is the useful one. It stands up a fake Tiiny, runs a story through the real
engine, and checks the event contract, the parent log, the media routes, the shelf, replay
and the transparency data. It needs no device and takes about two minutes.

---

## Credits

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="static/tiiny-logo.svg">
  <img alt="Tiiny" src="static/tiiny-logo-ink.svg" width="104">
</picture>

Built on the [Tiiny Pocket](https://tiiny.ai/) by Tiiny AI, whose PowerInfer and
TurboSparse work is the reason a device this size can run a 35B model at all.

Built by [Jason Brashear](https://github.com/webdevtodayjason).

Python standard library only. One HTML file per view. No dependencies, no build step, no
telemetry, no network egress beyond your own device.
