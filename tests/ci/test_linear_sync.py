import importlib.util
import re
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "linear_sync.py"
_spec = importlib.util.spec_from_file_location("linear_sync", _SCRIPT)
assert _spec and _spec.loader
ls = importlib.util.module_from_spec(_spec)
# Registered before exec so dataclasses can resolve the module's annotations.
sys.modules["linear_sync"] = ls
_spec.loader.exec_module(ls)

REPO = "datologyai/zephon"
RELEASE = "v0.3 Release: curricula"


class FakeWorld:
    """In-memory Linear workspace and GitHub repo behind a real ``_Client``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
        monkeypatch.setenv("LINEAR_API_KEY", "test")
        monkeypatch.setenv("GITHUB_TOKEN", "test")
        self.issues: dict[str, dict[str, Any]] = {}
        self.attachments: dict[str, str] = {}
        self.prs: dict[int, dict[str, Any]] = {}
        self.fail_next_attach = False
        self.client = ls._Client(dry_run=False)
        monkeypatch.setattr(self.client, "linear", self._linear)
        monkeypatch.setattr(self.client, "github", self._github)
        monkeypatch.setattr(self.client, "milestone_number", lambda title: title)

    def add_issue(
        self, identifier: str, milestone: str | None = None, team: str = "ZEPH"
    ) -> None:
        self.issues[identifier] = {
            "identifier": identifier,
            "team": team,
            "milestone": milestone,
            "description": "",
        }

    def add_pr(
        self,
        number: int,
        head: str,
        base: str = "main",
        body: str = "",
        milestone: str | None = None,
    ) -> dict[str, Any]:
        self.prs[number] = {
            "number": number,
            "title": f"PR {number}",
            "body": body,
            "head": {"ref": head},
            "base": {"ref": base},
            "html_url": f"https://github.com/{REPO}/pull/{number}",
            "user": {"login": "someone", "type": "User"},
            "draft": False,
            "milestone": {"title": milestone} if milestone else None,
        }
        return self.prs[number]

    def handle(self, number: int, create: bool = True) -> bool:
        return ls._handle(self.client, self.prs[number], "main", create)

    def _issue_node(self, identifier: str) -> dict[str, Any]:
        issue = self.issues[identifier]
        milestone = issue["milestone"]
        return {
            "id": identifier,
            "identifier": identifier,
            "team": {"key": issue["team"]},
            "projectMilestone": {"name": milestone} if milestone else None,
        }

    def _linear(self, query: str, **variables: Any) -> dict[str, Any]:
        if "issue(id:" in query:
            if variables["id"] not in self.issues:
                raise ls._LinearError([{"message": "Entity not found: Issue"}])
            return {"issue": self._issue_node(variables["id"])}
        if "attachmentsForURL" in query:
            owner = self.attachments.get(variables["url"])
            nodes = [{"issue": self._issue_node(owner)}] if owner else []
            return {"attachmentsForURL": {"nodes": nodes}}
        if "issues(" in query:
            nodes = [
                self._issue_node(i["identifier"])
                for i in self.issues.values()
                if variables["marker"] in i["description"]
            ]
            return {"issues": {"nodes": nodes[:1]}}
        if "teams(" in query:
            states = {"nodes": [{"id": "triage"}]}
            team = {"id": "zeph", "states": states, "labels": {"nodes": []}}
            return {"teams": {"nodes": [team]}}
        if "issueCreate" in query:
            identifier = f"ZEPH-{100 + len(self.issues)}"
            self.add_issue(identifier)
            self.issues[identifier]["description"] = variables["input"]["description"]
            return {
                "issueCreate": {"issue": {"id": identifier, "identifier": identifier}}
            }
        if "attachmentLinkGitHubPR" in query:
            if self.fail_next_attach:
                self.fail_next_attach = False
                raise urllib.error.URLError("timed out")
            if variables["url"] in self.attachments:
                raise ls._LinearError(
                    [{"message": "Duplicate attachment for duplicate url"}]
                )
            self.attachments[variables["url"]] = variables["issueId"]
            return {"attachmentLinkGitHubPR": {"success": True}}
        raise AssertionError(f"unexpected Linear query: {query}")

    def _github(
        self, method: str, path: str, data: dict[str, Any] | None = None
    ) -> Any:
        if method == "GET" and (match := re.search(r"head=[^:]+:(.+)$", path)):
            return [pr for pr in self.prs.values() if pr["head"]["ref"] == match[1]]
        if method == "PATCH" and (match := re.search(r"/issues/(\d+)$", path)):
            assert data is not None
            title = data["milestone"]
            self.prs[int(match[1])]["milestone"] = {"title": title} if title else None
            return {}
        raise AssertionError(f"unexpected GitHub call: {method} {path}")


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    return FakeWorld(monkeypatch)


def _milestone(world: FakeWorld, number: int) -> str | None:
    return (world.prs[number]["milestone"] or {}).get("title")


def test_magic_word_id_wins_over_plain_mention(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", RELEASE)
    world.add_issue("ZEPH-10")
    world.add_pr(1, "feature", body="Unlike ZEPH-2, this ...\n\nFixes ZEPH-10")
    assert world.handle(1)
    assert _milestone(world, 1) is None


def test_release_sets_milestone(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", RELEASE)
    world.add_pr(1, "feature", body="Fixes ZEPH-2")
    assert world.handle(1)
    assert _milestone(world, 1) == "v0.3"


def test_hand_set_milestone_is_never_replaced(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", RELEASE)
    world.add_pr(1, "feature", body="Fixes ZEPH-2", milestone="backlog")
    assert world.handle(1)
    assert _milestone(world, 1) == "backlog"


def test_plat_only_reference_fails(world: FakeWorld) -> None:
    world.add_pr(1, "feature", body="Fixes PLAT-42")
    assert not world.handle(1)


def test_issue_moved_out_of_zeph_is_ignored(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", RELEASE, team="PLAT")
    world.add_pr(1, "feature", body="Fixes ZEPH-2", milestone="v0.3")
    assert world.handle(1, create=False)
    assert _milestone(world, 1) is None


def test_transient_linear_error_keeps_milestone(
    world: FakeWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add_pr(1, "feature", body="Fixes ZEPH-2", milestone="v0.3")

    def rate_limited(query: str, **variables: Any) -> dict[str, Any]:
        raise ls._LinearError([{"message": "Rate limit exceeded"}])

    monkeypatch.setattr(world.client, "linear", rate_limited)
    with pytest.raises(ls._LinearError):
        world.handle(1, create=False)
    assert _milestone(world, 1) == "v0.3"


def test_stack_child_keeps_issue_after_parent_merges(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", RELEASE)
    world.add_pr(1, "parent", body="Contributes to ZEPH-2")
    world.add_pr(2, "child", base="parent")
    assert world.handle(2)
    assert _milestone(world, 2) == "v0.3"

    del world.prs[1]
    world.prs[2]["base"]["ref"] = "main"
    assert world.handle(2, create=False)
    assert world.handle(2)
    assert _milestone(world, 2) == "v0.3"
    assert set(world.issues) == {"ZEPH-2"}


def test_triage_retry_after_failed_attach_reuses_issue(world: FakeWorld) -> None:
    world.add_pr(1, "feature")
    world.fail_next_attach = True
    with pytest.raises(urllib.error.URLError):
        world.handle(1)
    assert world.handle(1)
    assert len(world.issues) == 1
    (identifier,) = world.issues
    assert world.attachments[world.prs[1]["html_url"]] == identifier
