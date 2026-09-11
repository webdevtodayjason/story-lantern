# Safety layer: design notes

Story Lantern's safety layer fails closed. If a page cannot be checked, the child does not
hear it. This note records how that default was arrived at and exactly what runs, so you can
audit it rather than take my word for it.

## Why fail closed

Three independent things run before a child hears a word: a deterministic backstop on the
request, a charter check on what was asked for, and a classifier pass on every generated
page. The interesting case is the third one being unavailable.

The device does one inference at a time, so a classifier call can be squeezed out under
contention, most plausibly by another app on the same Tiiny returning `150004`. An early
build showed the page anyway in that window and badged it amber in the parent log. The
reasoning was defensible: the other two checks had already passed, and refusing every page
during a contention window ends a child's story for a reason unrelated to content.

Review changed it, because the audience is a five-year-old at bedtime and "we logged it
loudly" helps nobody once the child has already heard it. The cost of failing closed turned
out to be near zero, because a warm pre-written closing page already existed for the block
case. So an unverifiable page now splices that ending instead. The story is slightly
shorter; it is never unverified.

Fail-open is still reachable at `LANTERN_ALLOW_UNVERIFIED=1`, as a knowing opt-in rather
than a default.

## What changed

`StorySession._judge` in `lantern.py`:

- Classifier unreachable now returns `UNVERIFIED`, which splices the bundled closing page.
  The child gets a slightly shorter story; they never get an unverified one. It gets its
  own page verdict (`UNVERIFIED_FALLBACK`) and its own paragraph in the parent log,
  because "two drafts failed" and "nothing could be checked" are different facts and the
  log used to report both as the first.
- A `CLASSIFIER_UNAVAILABLE` row still goes into `safety_event` with the page text verbatim,
  so the parent log shows exactly what was held back and why.
- Fail-open is still available, but it is now an explicit, knowing opt-in:
  `LANTERN_ALLOW_UNVERIFIED=1`. On that path the page is badged `ALLOW_UNVERIFIED` in the
  parent log, which already had the amber styling for it.

- A verdict the classifier returns that is NOT one of ALLOW/SOFTEN/BLOCK - a missing
  `verdict` key, or a synonym like UNSAFE or REJECT - used to default to `ALLOW`, which
  was the exact inverse of the policy on this page and of what `safety.py` had always
  done. It is now `SOFTEN`: one regeneration, then the fallback page.
- The page text is now delimited when it is handed to the classifier, with a standing
  instruction that nothing inside the markers is an instruction, and the classifier's
  reply is parsed from `message.content` only. Scavenging a safety verdict out of
  `reasoning_content` meant a JSON object the model quoted back while thinking could
  become the verdict.

Two related gaps were closed in the same pass:

- The deterministic backstop's `BLOCK` was being downgraded to `SOFTEN` for everything,
  which meant a page that produced gore got a polite "try again" instead of ending the
  story. Catastrophic categories (`gore`, `sexual`, `self_harm`, `real_violence`, `drugs`,
  `hate`, `abuse_disclosure`, `danger_howto`, `injection`, `body_horror`) now go straight to
  the fallback page. Regeneration is for a story that leaned tense, not for one that
  produced something that should never be regenerated toward.
- `safety.backstop_request`'s `story_steer` was computed and then dropped on the floor. It
  is now carried into both the plan prompt and every page prompt.
