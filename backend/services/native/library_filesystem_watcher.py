"""S-01 Hook C: zero-dependency recursive filesystem poller (D6).

The poller takes a recursive stat-only snapshot of every library root once
per poll interval and, when a snapshot moves, enqueues a single
``kind="incremental"`` / ``trigger="automatic"`` scan after a batching
window so rapid bursts (saves, multi-file copies, rename storms) collapse
into one ``request_run``. The request is scoped to the folders whose files
changed (a whole-root scan only for a root-level change, a bulk change, or a
scoping fault), so DroppedNeedle's own imports do not re-walk the library.

Recursion is load-bearing, not incidental: on Linux only the immediate
parent directory's mtime moves when a nested file changes, so a shallow
single-level poll misses nested mutations silently and permanently. The
walk is stat-only (no tag reads, no hashing) and runs in
``asyncio.to_thread`` so the event loop never blocks on directory I/O.

Known limitation (documented, accepted): changes that preserve both mtime
and size (``touch -r``, timestamp-preserving copies, same-size rewrites
inside the filesystem timestamp granularity) are invisible to a stat-only
snapshot and are picked up only by the rolling schedule or a manual scan.

Symlinks are never followed (``follow_symlinks=False`` throughout), so a
symlinked directory is recorded as a single entry and never descended.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path, PurePath
from typing import NamedTuple

from core.task_registry import TaskRegistry
from infrastructure.queue.durable_work_wakeup import DurableWorkWakeups
from models.library_work import ScanRequest, ScanScope
from services.local_files_service import AUDIO_EXTENSIONS
from services.native.library_filesystem_coordinator import is_management_artifact
from services.native.library_policy_resolver import (
    LibraryPolicyResolver,
    ResolvedLibraryPolicy,
)
from services.native.library_scan_coordinator import LibraryScanCoordinator
from services.native.library_scan_scheduler import LibraryAutomaticScanScheduler

logger = logging.getLogger(__name__)

WATCHER_TASK_NAME = "target-library-filesystem-watcher"

DEFAULT_POLL_INTERVAL_SECONDS = 300.0
DEFAULT_BATCH_WINDOW_SECONDS = 60.0
_MIN_POLL_INTERVAL_SECONDS = 1.0
# Past this many changed folders in one root (a bulk copy or restore) a folder
# per scope costs more than one walk of the root, so scan the whole root.
MAX_SCOPED_FOLDERS = 64

_Snapshot = dict[str, tuple[int, int, bool]]


class WatcherSettings(NamedTuple):
    """Duck-compatible with the prefs ``LibraryScanFilesystemWatcherSettings``
    section (``enabled`` / ``poll_interval_seconds`` / ``batch_window_seconds``)
    so the loop can run unconfigured with Hook C defaults."""

    enabled: bool = True
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    batch_window_seconds: float = DEFAULT_BATCH_WINDOW_SECONDS


_DEFAULT_SETTINGS = WatcherSettings()


def _snapshot_tree(root: Path) -> _Snapshot:
    """Recursive stat-only snapshot of ``root`` keyed by root-relative path.

    Values are ``(st_mtime_ns, st_size, is_dir)``. An unreadable subdirectory
    is skipped (the stale comparison simply retries next poll); a missing
    top-level root raises ``OSError`` to the caller, which logs and keeps the
    previous baseline instead of scan-storming on a transient unmount.
    """
    snapshot: _Snapshot = {}
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                entries = list(iterator)
        except OSError:
            if current == root:
                raise
            logger.debug("Filesystem watcher skipping unreadable directory: %s", current)
            continue
        for entry in entries:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            try:
                stat_result = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            try:
                relative = str(Path(entry.path).relative_to(root))
            except ValueError:  # pragma: no cover - scandir child outside root
                continue
            snapshot[relative] = (stat_result.st_mtime_ns, stat_result.st_size, is_dir)
            if is_dir:
                stack.append(Path(entry.path))
    return snapshot


def _is_scanned_file(relative: str, entry: tuple[int, int, bool]) -> bool:
    """Whether a snapshot entry is a file the inventory scanner would index."""
    return (
        not entry[2]
        and Path(relative).suffix.casefold() in AUDIO_EXTENSIONS
        and not is_management_artifact(Path(relative))
    )


def _changed_folders(previous: _Snapshot, current: _Snapshot) -> set[str]:
    """Folders (root-relative, ``"."`` for the root) holding a changed audio file.

    Only files the scanner indexes count (covers, logs, ``.DS_Store`` and
    management artifacts cannot change the catalogue). Directory entries are
    ignored: a directory's mtime moves whenever a child is added or removed,
    so counting it would turn a new artist folder into a whole-root scan. A
    file replaced by a directory of the same name counts as removed.
    """
    folders: set[str] = set()
    for relative, entry in current.items():
        if _is_scanned_file(relative, entry) and previous.get(relative) != entry:
            folders.add(str(PurePath(relative).parent))
    for relative, entry in previous.items():
        if not _is_scanned_file(relative, entry):
            continue
        kept = current.get(relative)
        if kept is None or kept[2]:
            folders.add(str(PurePath(relative).parent))
    return folders


def _scannable_folder(folder: str, snapshot: _Snapshot) -> str:
    """Nearest ancestor of ``folder`` (itself included) a scoped scan can use.

    A scope needs a directory that still exists, or the run would fail with
    ROOT_UNAVAILABLE, and an NFC path, because stored rows are NFC-normalised
    and reconciliation matches scopes by prefix. ``"."`` means whole root.
    """
    while folder != ".":
        entry = snapshot.get(folder)
        if (
            entry is not None
            and entry[2]
            and unicodedata.normalize("NFC", folder) == folder
        ):
            break
        folder = str(PurePath(folder).parent)
    return folder


def _collapse_nested(folders: set[str]) -> list[str]:
    """Drop folders that sit under another selected folder (keep the ancestor)."""
    return sorted(
        folder
        for folder in folders
        if not any(str(parent) in folders for parent in PurePath(folder).parents)
    )


def _scoped_scopes(
    root_id: str,
    folders: set[str],
    resolver: LibraryPolicyResolver,
    snapshot: _Snapshot,
) -> list[ScanScope] | None:
    """Per-folder scopes for one root, or ``None`` when it needs a whole-root scan."""
    usable = {_scannable_folder(folder, snapshot) for folder in folders}
    if "." in usable:
        return None
    root_paths = {root.id: root.path for root in resolver.settings.library_roots}
    root_path = root_paths.get(root_id)
    if root_path is None:
        return []
    # Resolve before collapsing: an excluded ancestor must not swallow an
    # included descendant (a rule can re-include a path under it). An included
    # ancestor may still cover an excluded descendant, as the scanner
    # re-resolves the policy of every file it finds.
    resolved_by_path: dict[str, ResolvedLibraryPolicy] = {}
    for folder in sorted(usable):
        resolved = resolver.resolve(Path(root_path) / folder)
        if resolved is None or resolved.policy == "excluded":
            continue
        resolved_by_path[resolved.relative_path] = resolved
    collapsed = _collapse_nested(set(resolved_by_path))
    if len(collapsed) > MAX_SCOPED_FOLDERS:
        return None
    return [
        ScanScope(
            root_id=resolved_by_path[path].root_id,
            scope_id=f"watch:{resolved_by_path[path].root_id}:{path}",
            relative_path=path,
            root_path=root_paths[resolved_by_path[path].root_id],
            effective_policy=resolved_by_path[path].policy,
            policy_revision=resolver.policy_revision,
        )
        for path in collapsed
    ]


def _pending_scopes(
    pending_folders: dict[str, set[str]],
    snapshots: dict[str, _Snapshot],
    resolver: LibraryPolicyResolver,
    scheduler: LibraryAutomaticScanScheduler,
) -> list[ScanScope]:
    """Scopes for the folders seen changing, falling back to whole roots."""
    scopes: list[ScanScope] = []
    for root_id, folders in pending_folders.items():
        try:
            scoped = _scoped_scopes(root_id, folders, resolver, snapshots[root_id])
        except Exception:  # noqa: BLE001 - any scoping fault falls back to the whole root, as before folder scoping
            logger.debug(
                "Filesystem watcher could not scope root %s; scanning it whole",
                root_id,
                exc_info=True,
            )
            scoped = None
        if scoped is None:
            scoped = [
                scope
                for scope in scheduler.scheduled_scopes(resolver)
                if scope.root_id == root_id
            ]
        scopes.extend(scoped)
    return scopes


def _log_watcher_error(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "Target filesystem watcher stopped unexpectedly",
            exc_info=(type(error), error, error.__traceback__),
        )


def start_library_filesystem_watcher(
    coordinator_getter: Callable[[], LibraryScanCoordinator],
    root_paths_getter: Callable[[], dict[str, Path]],
    work_wakeups: DurableWorkWakeups,
    *,
    scheduler_getter: Callable[[], LibraryAutomaticScanScheduler] | None = None,
    resolver_getter: Callable[[], LibraryPolicyResolver] | None = None,
    watcher_settings_getter: Callable[[], WatcherSettings] | None = None,
    clock_getter: Callable[[], float] = time.monotonic,
) -> asyncio.Task[None]:
    registry = TaskRegistry.get_instance()
    if registry.is_running(WATCHER_TASK_NAME):
        raise RuntimeError(f"Task '{WATCHER_TASK_NAME}' is already running")
    task = asyncio.create_task(
        watch_library_filesystem(
            coordinator_getter,
            root_paths_getter,
            work_wakeups,
            scheduler_getter=scheduler_getter,
            resolver_getter=resolver_getter,
            watcher_settings_getter=watcher_settings_getter,
            clock_getter=clock_getter,
        )
    )
    try:
        registry.register(WATCHER_TASK_NAME, task)
    except RuntimeError:
        # R-05: single-loop double-start is impossible (no await sits between
        # the is_running check above and register), so a register conflict
        # here means threads/reentrancy. Cancel the orphan just created - the
        # already-registered task keeps running - and re-raise. This starter
        # is synchronous, so the orphan cannot be awaited here; cancel plus
        # the standard error-logging done-callback consumes its outcome (a
        # cancelled task carries no retrievable exception, and the callback
        # returns early for cancelled tasks).
        task.cancel()
        task.add_done_callback(_log_watcher_error)
        raise
    task.add_done_callback(_log_watcher_error)
    return task


async def watch_library_filesystem(
    coordinator_getter: Callable[[], LibraryScanCoordinator],
    root_paths_getter: Callable[[], dict[str, Path]],
    work_wakeups: DurableWorkWakeups | None = None,
    *,
    scheduler_getter: Callable[[], LibraryAutomaticScanScheduler] | None = None,
    resolver_getter: Callable[[], LibraryPolicyResolver] | None = None,
    watcher_settings_getter: Callable[[], WatcherSettings] | None = None,
    clock_getter: Callable[[], float] = time.monotonic,
) -> None:
    """Poll library roots for filesystem mutations and batch them into scans.

    Every getter is re-read each iteration (never captured): a settings save
    rebuilds singletons, and the next poll must see the new instances.
    """
    wakeups = work_wakeups or DurableWorkWakeups()
    settings_getter = watcher_settings_getter or (lambda: _DEFAULT_SETTINGS)
    baselines: dict[str, tuple[str, _Snapshot]] = {}
    # Folders seen changing per root since the last request, kept across polls
    # so a burst spread over several polls inside the window loses none.
    pending_folders: dict[str, set[str]] = {}
    pending_since: float | None = None
    while True:
        sleep_seconds = DEFAULT_POLL_INTERVAL_SECONDS
        try:
            settings = settings_getter()
            poll_interval = max(float(settings.poll_interval_seconds), _MIN_POLL_INTERVAL_SECONDS)
            window = max(float(settings.batch_window_seconds), 0.0)
            sleep_seconds = poll_interval
            coordinator = coordinator_getter()
            resolver = resolver_getter() if resolver_getter is not None else None
            enabled = bool(settings.enabled) and (
                resolver is None or resolver.settings.enabled
            )
            if (
                enabled
                and resolver is not None
                and scheduler_getter is not None
            ):
                now = clock_getter()
                for root_id, root_path in root_paths_getter().items():
                    snapshot = await asyncio.to_thread(_snapshot_tree, root_path)
                    previous = baselines.get(root_id)
                    if previous is None or previous[0] != str(root_path):
                        # First sighting (or a re-pointed root id): seed the
                        # baseline silently so a (re)start never scans a
                        # steady tree. Settings-driven root changes are
                        # already covered by Hook B dirty marks.
                        baselines[root_id] = (str(root_path), snapshot)
                        pending_folders.pop(root_id, None)
                        if not pending_folders:
                            pending_since = None
                        continue
                    if snapshot != previous[1]:
                        baselines[root_id] = (str(root_path), snapshot)
                        changed = _changed_folders(previous[1], snapshot)
                        if changed:
                            pending_folders.setdefault(root_id, set()).update(changed)
                            if pending_since is None:
                                pending_since = now
                if pending_since is not None:
                    elapsed = clock_getter() - pending_since
                    if elapsed >= window:
                        # Off the loop: resolving a policy touches the disk.
                        scopes = await asyncio.to_thread(
                            _pending_scopes,
                            pending_folders,
                            {
                                root_id: baseline[1]
                                for root_id, baseline in baselines.items()
                            },
                            resolver,
                            scheduler_getter(),
                        )
                        if not scopes:
                            logger.debug(
                                "Filesystem watcher dropping pending scan: no scannable scopes"
                            )
                            pending_since = None
                            pending_folders.clear()
                        else:
                            result = await coordinator.request_run(
                                ScanRequest(
                                    kind="incremental",
                                    trigger="automatic",
                                    policy_revision=resolver.policy_revision,
                                    scopes=scopes,
                                )
                            )
                            logger.info(
                                "Filesystem watcher requested incremental scan "
                                "(scopes=%s, disposition=%s)",
                                len(scopes),
                                result.disposition,
                            )
                            # Like the supervisor's dirty marks: a conflict
                            # keeps the pending folders for the next poll.
                            if result.disposition != "conflict":
                                pending_since = None
                                pending_folders.clear()
                                wakeups.notify("scan")
                    else:
                        sleep_seconds = min(poll_interval, window - elapsed)
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001 - the lifetime watcher records and survives iteration failures
            logger.exception("Target filesystem watcher iteration failed")
        try:
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            break
