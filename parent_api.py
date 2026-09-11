#!/usr/bin/env python3
"""
Story Lantern - parent_api.py

Two functions the parent page needs and the engine does not have yet:

    parent_data(conn, child=None, ...)  -> the aggregate JSON for /api/parent/data
    delete_story(conn, story_id, ...)   -> actually remove one story and its media

Kept out of lantern.py deliberately: deleting a family's data and building the
transparency view are the parts of this product that should be small, readable, and
testable on their own, not buried in a 1900-line server.

Wiring it up is three lines in lantern.py's Handler:

    import parent_api                                            # top of file

    # in do_GET, next to the other /api routes:
    if path == "/api/parent/data":
        return self._json(parent_api.parent_data(
            db(), child={"name": L.cfg.child_name, "age": L.cfg.child_age},
            charter_version=getattr(L.cfg, "charter_version", None)))

    # in do_POST:
    m = re.fullmatch(r"/api/story/(\\d+)/delete", path)
    if m:
        return self._json(parent_api.delete_story(db(), int(m.group(1)),
                                                  media_root=L.cfg.media_dir))

Python 3.11+, standard library only.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

__all__ = ["parent_data", "delete_story", "media_usage"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))
    except sqlite3.Error:
        return False


def media_usage(media_root: Path | str | None) -> dict:
    """Bytes on disk under the media root. The parent page shows this because a full
    disk at month nine is the unglamorous way this product dies."""
    if not media_root:
        return {"media_mb": None}
    root = Path(media_root)
    total = 0
    try:
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        return {"media_mb": None}
    return {"media_mb": round(total / (1024 * 1024), 1)}


def parent_data(conn: sqlite3.Connection, *, child: Optional[dict] = None,
                charter_version: Optional[str] = None,
                media_root: Path | str | None = None,
                limit: int = 60, residency_warning: Optional[str] = None) -> dict:
    """
    Everything the parent page renders, in one call.

    Nothing here is filtered or softened: `offending_text` comes out of the database
    exactly as it went in. If the model wrote something ugly and the safety layer caught
    it, the parent is entitled to read the ugly thing.
    """
    stories = _rows(conn, "SELECT * FROM story ORDER BY id DESC LIMIT ?", (limit,))
    out = []
    for s in stories:
        sid = s["id"]
        pages = _rows(conn, "SELECT * FROM page WHERE story_id=? ORDER BY idx", (sid,))
        events = _rows(conn, "SELECT * FROM safety_event WHERE story_id=? ORDER BY id",
                       (sid,))
        out.append({
            "id": sid,
            "created_at": s.get("created_at"),
            "title": s.get("title"),
            "status": s.get("status"),
            "child_name": (child or {}).get("name"),
            "raw_transcript": s.get("raw_transcript") or "",
            "corrected_transcript": s.get("corrected_transcript") or "",
            "redirect_note": _redirect_note(s.get("theme_contract")),
            "page_count": s.get("page_count"),
            "pages": [{
                "idx": p.get("idx"),
                "text": p.get("text") or "",
                "image_url": f"/page/{p['id']}.png" if p.get("image_path") else None,
                "audio_url": f"/page/{p['id']}.mp3" if p.get("audio_path") else None,
                "safety_verdict": p.get("safety_verdict") or "ALLOW",
                "safety_reason": p.get("safety_reason") or "",
                "regen_count": p.get("regen_count") or 0,
                "gen_ms": p.get("gen_ms") or 0,
            } for p in pages],
            "safety_events": events,
        })
    data = {
        "generated_at": _now(),
        "child": child,
        "charter_version": charter_version,
        "storage": media_usage(media_root),
        "stories": out,
    }
    if residency_warning:
        data["residency_warning"] = residency_warning
    return data


def _redirect_note(theme_contract_json: Optional[str]) -> Optional[str]:
    if not theme_contract_json:
        return None
    try:
        import json
        return (json.loads(theme_contract_json) or {}).get("redirect_note")
    except (ValueError, TypeError):
        return None


def delete_story(conn: sqlite3.Connection, story_id: int, *,
                 media_root: Path | str | None = None,
                 tombstone: bool = True) -> dict:
    """
    Remove one story: its pages, its illustrations, its narration, and its safety events.

    Two things this deliberately does NOT do:

    * It never touches the `character` table. Deleting last Tuesday's story must not
      delete Biscuit - the character bible is the product, and a parent tidying up the
      log would be very surprised to find the dog gone. The confirmation dialog on the
      parent page promises this.
    * It never deletes a file that is not underneath `media_root`. Paths come out of the
      database, and a path in a database is an instruction from somewhere else; a
      malformed or hostile `image_path` must not turn a tidy-up into an rm of /etc.
    """
    root = Path(media_root).resolve() if media_root else None
    pages = _rows(conn, "SELECT id,image_path,audio_path FROM page WHERE story_id=?",
                  (story_id,))
    if not pages and not _rows(conn, "SELECT id FROM story WHERE id=?", (story_id,)):
        return {"ok": False, "error": "no such story", "story_id": story_id}

    removed, skipped = 0, []
    for p in pages:
        for key in ("image_path", "audio_path"):
            raw = p.get(key)
            if not raw:
                continue
            try:
                f = Path(raw).resolve()
            except OSError:
                skipped.append(raw)
                continue
            if root is not None and root not in f.parents:
                skipped.append(str(f))       # outside the media root: leave it alone
                continue
            try:
                if f.is_file():
                    f.unlink()
                    removed += 1
            except OSError:
                skipped.append(str(f))

    # Take the story's media folder with it if it is now empty.
    if root is not None:
        for cand in (root / str(story_id), root / f"story_{story_id}"):
            try:
                if cand.is_dir() and not any(cand.iterdir()):
                    cand.rmdir()
            except OSError:
                pass

    n_pages = conn.execute("DELETE FROM page WHERE story_id=?", (story_id,)).rowcount
    n_events = conn.execute("DELETE FROM safety_event WHERE story_id=?",
                            (story_id,)).rowcount
    conn.execute("DELETE FROM story WHERE id=?", (story_id,))

    if tombstone:
        # A row saying a story was deleted, and NOTHING of what it said. The point of
        # the tombstone is that the log cannot quietly lose entries; it is not a way to
        # keep content a parent asked to be rid of.
        try:
            conn.execute(
                "INSERT INTO safety_event(story_id,page_idx,stage,verdict,reason,"
                "offending_text,at) VALUES (NULL,NULL,?,?,?,?,?)",
                ("deletion", "DELETED",
                 f"parent deleted story {story_id}: {n_pages} pages, {n_events} events, "
                 f"{removed} media files", "", _now()))
        except sqlite3.Error:
            pass
    conn.commit()
    return {"ok": True, "story_id": story_id, "pages": n_pages,
            "safety_events": n_events, "media_files": removed,
            "skipped_files": skipped}


# --------------------------------------------------------------------------------------

def _selftest() -> int:
    import json
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="lantern-parentapi-"))
    media = tmp / "media" / "3"
    media.mkdir(parents=True)
    img = media / "p0.png"
    img.write_bytes(b"\x89PNG")
    outside = tmp / "not-media.png"
    outside.write_bytes(b"\x89PNG")

    conn = sqlite3.connect(":memory:")
    conn.executescript("""
      CREATE TABLE story(id INTEGER PRIMARY KEY, created_at TEXT, raw_transcript TEXT,
        corrected_transcript TEXT, theme_contract TEXT, title TEXT, spine TEXT,
        story_seed INTEGER, page_count INTEGER, status TEXT, finished_at TEXT);
      CREATE TABLE page(id INTEGER PRIMARY KEY, story_id INTEGER, idx INTEGER, text TEXT,
        image_prompt TEXT, image_path TEXT, audio_path TEXT, character_ids TEXT,
        safety_verdict TEXT, safety_reason TEXT, regen_count INTEGER, gen_ms INTEGER);
      CREATE TABLE safety_event(id INTEGER PRIMARY KEY, story_id INTEGER, page_idx INTEGER,
        stage TEXT, verdict TEXT, reason TEXT, offending_text TEXT, at TEXT);
      CREATE TABLE character(id INTEGER PRIMARY KEY, name TEXT);
    """)
    conn.execute("INSERT INTO story VALUES (3,'2026-08-21T19:34:00Z','a story about zombies',"
                 "'a story about zombies','{\"redirect_note\":\"a polite skeleton\"}',"
                 "'The Polite Skeleton','[]',1,2,'finished',NULL)")
    conn.execute("INSERT INTO page VALUES (10,3,0,'page one','prompt',?,NULL,'[]',"
                 "'ALLOW','clean',0,10200)", (str(img),))
    conn.execute("INSERT INTO page VALUES (11,3,1,'page two','prompt',?,NULL,'[]',"
                 "'SOFTENED','regenerated once',1,19800)", (str(outside),))
    conn.execute("INSERT INTO safety_event VALUES (1,3,1,'page_text','BLOCK','injury',"
                 "'there was blood on the snow','2026-08-21T19:35:40Z')")
    conn.execute("INSERT INTO character VALUES (1,'Biscuit')")
    conn.commit()

    d = parent_data(conn, child={"name": "Maya", "age": 5},
                    charter_version="lantern-charter-1.0.0", media_root=tmp / "media")
    s = d["stories"][0]
    print("parent_data:")
    print(f"  stories={len(d['stories'])} pages={len(s['pages'])} "
          f"events={len(s['safety_events'])} media_mb={d['storage']['media_mb']}")
    print(f"  redirect_note={s['redirect_note']!r}")
    print(f"  page0 image_url={s['pages'][0]['image_url']}")
    print(f"  verbatim event text kept: "
          f"{s['safety_events'][0]['offending_text']!r}")
    assert s["redirect_note"] == "a polite skeleton"
    assert s["pages"][0]["image_url"] == "/page/10.png"
    assert s["safety_events"][0]["offending_text"] == "there was blood on the snow"
    assert json.dumps(d)  # must be serialisable as-is

    r = delete_story(conn, 3, media_root=tmp / "media")
    print("\ndelete_story:", r)
    assert r["ok"] and r["pages"] == 2 and r["media_files"] == 1
    assert not img.exists(), "media file inside the root should be gone"
    assert outside.exists(), "a path outside media_root must NOT be deleted"
    assert str(outside.resolve()) in r["skipped_files"]
    assert not media.exists(), "empty story media folder should be removed"
    assert conn.execute("SELECT COUNT(*) FROM story").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM character").fetchone()[0] == 1, \
        "deleting a story must never delete a character"
    tomb = conn.execute("SELECT verdict,reason,offending_text FROM safety_event").fetchall()
    print("tombstone:", tomb)
    assert len(tomb) == 1 and tomb[0][0] == "DELETED" and tomb[0][2] == ""

    print("\nmissing story:", delete_story(conn, 999, media_root=tmp / "media"))
    print("\nall assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
