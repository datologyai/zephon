import importlib.util
import io
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
INBOX = "INBOX"


def _as_linear_stores(markdown: str) -> str:
    # Linear rewrites a bare URL in a description to [url](<url>).
    return re.sub(r"(?<![\[<])(https://[^\s\])>]+)", r"[\1](<\1>)", markdown)


class FakeWorld:
    """In-memory Linear workspace and GitHub repo behind a real ``_Client``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
        monkeypatch.setenv("LINEAR_API_KEY", "test")
        monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "test")
        monkeypatch.setenv("LINEAR_TRIAGE_TEAM", INBOX)
        self.issues: dict[str, dict[str, Any]] = {}
        # PR or GitHub issue URL -> (issue identifier, link kind)
        self.attachments: dict[str, tuple[str, str]] = {}
        self.prs: dict[int, dict[str, Any]] = {}
        self.comments: dict[int, list[str]] = {}
        # Teams the credentials can read; issues in other teams are not found.
        self.visible_teams = {"ZEPH", INBOX}
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
            owner, kind = self.attachments.get(variables["url"], (None, None))
            nodes = []
            if owner and self._visible(owner):
                nodes.append(
                    {
                        "id": f"att:{variables['url']}",
                        "metadata": {"linkKind": kind},
                        "issue": self._issue_node(owner),
                    }
                )
            return {"attachmentsForURL": {"nodes": nodes}}
        if "attachmentDelete" in query:
            del self.attachments[variables["id"].removeprefix("att:")]
            return {"attachmentDelete": {"success": True}}
        if "issues(" in query:
            nodes = [
                self._issue_node(key)
                for key, i in self.issues.items()
                if i["team"] in variables["keys"]
                and self._visible(key)
                and variables["marker"] in i["description"]
            ]
            return {"issues": {"nodes": nodes[:1]}}
        if "teams(" in query:
            states = {"nodes": [{"id": "triage"}]}
            team = {"id": variables["key"], "states": states, "labels": {"nodes": []}}
            return {"teams": {"nodes": [team]}}
        if "issueCreate" in query:
            team = variables["input"]["teamId"]
            identifier = f"{team}-{100 + len(self.issues)}"
            self.add_issue(identifier, team=team)
            self.issues[identifier]["description"] = _as_linear_stores(
                variables["input"]["description"]
            )
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
            self.attachments[variables["url"]] = (
                variables["issueId"],
                variables["kind"],
            )
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
    assert world.attachments[world.prs[1]["html_url"]] == (identifier, "closes")


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
        f"{ls.TRIAGE_COMMENT_MARKER}\nTracked in Linear for triage."
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
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
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


def test_hand_attached_stack_child_keeps_its_issue(world: FakeWorld) -> None:
    world.add_issue("ZEPH-2", RELEASE)
    world.add_issue("ZEPH-5", "v0.4 Release: next")
    world.add_pr(1, "parent", body="Contributes to ZEPH-2")
    child = world.add_pr(2, "child", base="parent")
    # Attaching by hand in Linear's UI creates a "links" attachment.
    world.attachments[child["html_url"]] = ("ZEPH-5", "links")
    assert world.handle(2)
    assert world.attachments[child["html_url"]] == ("ZEPH-5", "links")
    assert _milestone(world, 2) == "v0.4"


def test_triage_issue_is_opened_in_the_inbox_team(world: FakeWorld) -> None:
    world.add_pr(1, "feature")
    assert world.handle(1)
    (identifier,) = world.issues
    assert world.issues[identifier]["team"] == INBOX
    assert world.issues[identifier]["description"] == (
        f"Opened from [{world.prs[1]['html_url']}](<{world.prs[1]['html_url']}>) "
        "by @someone."
    )
    assert world.handle(1)
    assert set(world.issues) == {identifier}


def test_no_triage_issue_without_triage_team(
    world: FakeWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(world.client, "triage_team", None)
    world.add_pr(1, "feature")
    assert world.handle(1)
    assert world.issues == {}
    assert 1 not in world.comments


def test_accepted_triage_issue_keeps_its_pr(world: FakeWorld) -> None:
    world.add_pr(1, "feature")
    assert world.handle(1)
    (identifier,) = world.issues
    world.move(identifier, "ZEPH")
    world.issues[identifier]["milestone"] = RELEASE
    assert world.handle(1)
    assert set(world.issues) == {identifier}
    assert _milestone(world, 1) == "v0.3"


def test_fixes_synced_github_issue_uses_its_zeph_issue(world: FakeWorld) -> None:
    world.add_issue("ZEPH-20", RELEASE)
    world.attachments[f"https://github.com/{REPO}/issues/86"] = ("ZEPH-20", "links")
    world.add_pr(1, "feature", body="Fixes #86")
    assert world.handle(1)
    assert _milestone(world, 1) == "v0.3"
    assert set(world.issues) == {"ZEPH-20"}


@pytest.mark.parametrize(
    "body",
    [
        "Closes datologyai/zephon#86",
        f"Resolves https://github.com/{REPO}/issues/86",
        "fixes: #86",
    ],
)
def test_github_issue_reference_forms(body: str) -> None:
    pr = {"title": "t", "body": body}
    assert ls._referenced_github_issues(pr, REPO) == [86]


def test_plain_hash_mention_is_not_a_github_issue_link() -> None:
    pr = {"title": "t", "body": "Follows up on #86 and other/repo#5."}
    assert ls._referenced_github_issues(pr, REPO) == []


def test_fixes_unsynced_github_issue_still_gets_triage(world: FakeWorld) -> None:
    world.add_pr(1, "feature", body="Fixes #49")
    assert world.handle(1)
    (identifier,) = world.issues
    assert world.issues[identifier]["team"] == INBOX


def test_plat_reference_next_to_unsynced_github_issue_fails(
    world: FakeWorld,
) -> None:
    world.add_pr(1, "feature", body="Fixes #49\n\nRelated to PLAT-4")
    assert not world.handle(1, create=False)


def test_plat_reference_next_to_synced_github_issue_passes(
    world: FakeWorld,
) -> None:
    world.add_issue("ZEPH-20")
    world.attachments[f"https://github.com/{REPO}/issues/86"] = ("ZEPH-20", "links")
    world.add_pr(1, "feature", body="Fixes #86\n\nRelated to PLAT-4")
    assert world.handle(1, create=False)


def test_oauth_app_credentials_are_exchanged_for_a_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINEAR_CLIENT_ID", "id")
    monkeypatch.setenv("LINEAR_CLIENT_SECRET", "secret")
    sent: dict[str, Any] = {}

    def urlopen(url: str, data: bytes) -> Any:
        sent.update(url=url, form=urllib.parse.parse_qs(data.decode()))
        return io.BytesIO(b'{"access_token": "tok"}')

    monkeypatch.setattr(ls.urllib.request, "urlopen", urlopen)
    assert ls._linear_auth() == "Bearer tok"
    assert sent["url"] == ls.LINEAR_TOKEN_URL
    assert sent["form"]["grant_type"] == ["client_credentials"]
    assert sent["form"]["client_secret"] == ["secret"]


def test_personal_api_key_is_used_without_app_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LINEAR_CLIENT_ID", raising=False)
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_x")
    assert ls._linear_auth() == "lin_api_x"
