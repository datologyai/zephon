import importlib.util
import re
import sys
import urllib.error
import urllib.parse
from email.message import Message
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
        self.comments: dict[int, list[str]] = {}
        # Like the real key, which is scoped to ZEPH: other teams are invisible.
        self.visible_teams = {"ZEPH"}
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

    def add_bot_pr(self, number: int, head: str) -> dict[str, Any]:
        pr = self.add_pr(number, head)
        pr["user"] = {"login": "dependabot[bot]", "type": "Bot"}
        return pr

    def handle(self, number: int, create: bool = True) -> bool:
        return ls._handle(self.client, self.prs[number], "main", create)

    def move(self, identifier: str, team: str) -> None:
        self.issues[identifier]["team"] = team

    def _visible(self, identifier: str) -> bool:
        return self.issues[identifier]["team"] in self.visible_teams

    def _issue_node(self, identifier: str) -> dict[str, Any]:
        issue = self.issues[identifier]
        milestone = issue["milestone"]
        return {
            "id": identifier,
            "identifier": issue["identifier"],
            "team": {"key": issue["team"]},
            "projectMilestone": {"name": milestone} if milestone else None,
        }

    def _linear(self, query: str, **variables: Any) -> dict[str, Any]:
        if "issue(id:" in query:
            identifier = variables["id"]
            if identifier not in self.issues or not self._visible(identifier):
                raise ls._LinearError([{"message": "Entity not found: Issue"}])
            return {"issue": self._issue_node(identifier)}
        if "attachmentsForURL" in query:
            owner = self.attachments.get(variables["url"])
            nodes = []
            if owner and self._visible(owner):
                nodes.append(
                    {"id": f"att:{variables['url']}", "issue": self._issue_node(owner)}
                )
            return {"attachmentsForURL": {"nodes": nodes}}
        if "attachmentDelete" in query:
            del self.attachments[variables["id"].removeprefix("att:")]
            return {"attachmentDelete": {"success": True}}
        if "issues(" in query:
            nodes = [
                self._issue_node(key)
                for key, i in self.issues.items()
                if i["team"] == variables["key"]
                and self._visible(key)
                and variables["marker"] in i["description"]
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
            assert "&" not in match[1], "head must be URL-encoded"
            base = urllib.parse.unquote(match[1])
            return [pr for pr in self.prs.values() if pr["head"]["ref"] == base]
        if match := re.search(r"/issues/(\d+)/comments", path):
            comments = self.comments.setdefault(int(match[1]), [])
            if method == "POST":
                assert data is not None
                comments.append(data["body"])
                return {}
            return [{"body": body} for body in comments]
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
    world.visible_teams.add("PLAT")
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


def test_stack_parent_with_special_chars_in_branch(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", RELEASE)
    world.add_pr(1, "feature&cleanup", body="Contributes to ZEPH-2")
    world.add_pr(2, "child", base="feature&cleanup")
    assert world.handle(2)
    assert _milestone(world, 2) == "v0.3"


def test_plat_reference_fails_despite_zeph_attachment(world: FakeWorld) -> None:
    world.add_pr(1, "feature")
    assert world.handle(1)
    world.prs[1]["body"] = "Fixes PLAT-42"
    assert not world.handle(1)


def test_unresolved_zeph_id_creates_no_triage_issue(world: FakeWorld) -> None:
    world.add_pr(1, "feature", body="Fixes ZEPH-999")
    assert world.handle(1)
    assert world.issues == {}


def test_bot_pr_without_id_never_calls_linear(
    world: FakeWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add_bot_pr(1, "dependabot/pip/foo")

    def no_credentials(query: str, **variables: Any) -> dict[str, Any]:
        raise AssertionError("bot PR reached Linear")

    monkeypatch.setattr(world.client, "linear", no_credentials)
    assert world.handle(1)


def test_moved_issue_log_hides_new_identifier(
    world: FakeWorld, capsys: pytest.CaptureFixture[str]
) -> None:
    world.visible_teams.add("PLAT")
    world.add_issue("ZEPH-2", RELEASE, team="PLAT")
    world.issues["ZEPH-2"]["identifier"] = "PLAT-123"
    world.add_pr(1, "feature", body="Fixes ZEPH-2")
    assert world.handle(1)
    assert "PLAT-123" not in capsys.readouterr().out


def test_stack_child_follows_parent_switching_issue(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", RELEASE)
    world.add_issue("ZEPH-3", "v0.4 Release: next")
    world.add_pr(1, "parent", body="Contributes to ZEPH-2")
    world.add_pr(2, "child", base="parent")
    assert world.handle(2)
    assert _milestone(world, 2) == "v0.3"

    world.prs[1]["body"] = "Contributes to ZEPH-3"
    assert world.handle(2, create=False)
    assert _milestone(world, 2) == "v0.4"

    del world.prs[1]
    world.prs[2]["base"]["ref"] = "main"
    assert world.handle(2, create=False)
    assert _milestone(world, 2) == "v0.4"


def test_triage_issue_moved_out_of_zeph_is_not_reopened(world: FakeWorld) -> None:
    world.add_pr(1, "feature")
    assert world.handle(1)
    (identifier,) = world.issues
    assert world.comments[1] == [
        f"{ls.TRIAGE_COMMENT_MARKER}\nTracked in Linear as {identifier}."
    ]

    world.move(identifier, "PLAT")
    assert world.handle(1)
    assert set(world.issues) == {identifier}
    assert len(world.comments[1]) == 1


def test_stack_child_never_opens_its_own_triage_issue(world: FakeWorld) -> None:
    world.add_bot_pr(1, "parent")
    world.add_pr(2, "child", base="parent")
    assert world.handle(2)
    assert world.issues == {}


@pytest.mark.parametrize(
    ("name", "release"),
    [
        ("v0.3 Release: curricula", "v0.3"),
        ("v0.3.1 Hotfix", "v0.3.1"),
        ("Backlog", None),
    ],
)
def test_release_of_linear_milestone(name: str, release: str | None) -> None:
    assert ls._release_of({"name": name}) == release


def test_patch_release_replaces_minor_milestone(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", "v0.3.1 Hotfix")
    world.add_pr(1, "feature", body="Fixes ZEPH-2", milestone="v0.3")
    assert world.handle(1)
    assert _milestone(world, 1) == "v0.3.1"


def _milestone_client(monkeypatch: pytest.MonkeyPatch, created_by_race: bool) -> Any:
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("LINEAR_API_KEY", "test")
    monkeypatch.setenv("GITHUB_TOKEN", "test")
    client = ls._Client(dry_run=False)
    posts = 0

    def github(method: str, path: str, data: Any = None) -> Any:
        nonlocal posts
        if method == "GET":
            raced = created_by_race and posts > 0
            return [{"title": "v0.3", "number": 7}] if raced else []
        posts += 1
        raise urllib.error.HTTPError(path, 422, "Unprocessable", Message(), None)

    monkeypatch.setattr(client, "github", github)
    return client


def test_milestone_created_concurrently_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _milestone_client(monkeypatch, created_by_race=True)
    assert client.milestone_number("v0.3") == 7


def test_milestone_validation_error_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _milestone_client(monkeypatch, created_by_race=False)
    with pytest.raises(urllib.error.HTTPError):
        client.milestone_number("v0.3")
