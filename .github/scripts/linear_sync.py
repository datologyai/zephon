"""Keep GitHub PR milestones in step with their ZEPH issues in Linear.

Linear is the source of truth for release planning. A PR inherits the release of
the ZEPH issue it references: a Linear milestone named ``vX.Y ...`` or
``vX.Y.Z ...`` becomes the GitHub milestone ``vX.Y`` or ``vX.Y.Z``. Only that
version string is ever written to GitHub. A PR that references a GitHub issue
synced to ZEPH (``Fixes #12``) uses that issue.

An unlinked PR gets a triage issue in the team named by ``LINEAR_TRIAGE_TEAM``,
which must not be issue-synced to GitHub: an issue created in a two-way synced
team is copied to GitHub at once, duplicating the PR. Moving it into ZEPH once
accepted creates no copy. Without ``LINEAR_TRIAGE_TEAM`` no triage issues are
created.

Modes:
  pr         Handle the PR in ``$GITHUB_EVENT_PATH`` (pull_request_target).
  reconcile  Re-check every open PR (nightly), catching issues that moved
             between releases or left triage.

Pass ``--dry-run`` to log intended changes without writing anything.
Requires ``GITHUB_TOKEN``, ``GITHUB_REPOSITORY`` and Linear credentials: an
OAuth app's ``LINEAR_CLIENT_ID`` and ``LINEAR_CLIENT_SECRET``, or else
``LINEAR_API_KEY``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

LINEAR_URL = "https://api.linear.app/graphql"
LINEAR_TOKEN_URL = "https://api.linear.app/oauth/token"
GITHUB_API = "https://api.github.com"
TEAM_KEY = "ZEPH"
TRIAGE_LABEL = "from-github-pr"

ISSUE_ID_RE = re.compile(r"\b(ZEPH|PLAT)-(\d+)\b", re.IGNORECASE)
# Linear's magic words (a superset of GitHub's closing keywords); an issue after
# one of these is a link, not a mention.
MAGIC_WORDS = (
    r"\b(?:close[sd]?|closing|fix(?:e[sd])?|fixing|resolve[sd]?|resolving"
    r"|complete[sd]?|completing|refs?|references|part of|related to"
    r"|contributes to|towards?)\s*:?\s+"
)
MAGIC_ID_RE = re.compile(MAGIC_WORDS + r"((?:ZEPH|PLAT)-\d+)\b", re.IGNORECASE)
RELEASE_RE = re.compile(r"^v(\d+\.\d+(?:\.\d+)?)\b")
# Only milestones this script could have set are ever replaced or cleared, so a
# milestone a maintainer set by hand for another reason is left alone.
MANAGED_MILESTONE_RE = re.compile(r"^v\d+\.\d+(?:\.\d+)?$")
# Marks the PR comment recording a triage issue. The key may not see the issue
# once triage moves it to another team, so GitHub keeps the record. The comment
# names no issue: the identifier changes when the issue moves to ZEPH.
TRIAGE_COMMENT_MARKER = "<!-- linear-sync:triage -->"
# Link kind of the attachments this script adds for inherited issues. Linear only
# creates "contributes" links itself from a magic word, which needs an ID in the
# PR, and attaching by hand in Linear creates "links".
INHERITED_LINK_KIND = "contributes"


class _LinearError(RuntimeError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__(f"Linear API error: {errors}")
        self.errors = errors

    @property
    def not_found(self) -> bool:
        return self._all_start_with("Entity not found")

    @property
    def duplicate_attachment(self) -> bool:
        return self._all_start_with("Duplicate attachment")

    def _all_start_with(self, prefix: str) -> bool:
        return all(e.get("message", "").startswith(prefix) for e in self.errors)


def _linear_auth() -> str:
    client_id = os.environ.get("LINEAR_CLIENT_ID")
    if not client_id:
        return os.environ["LINEAR_API_KEY"]
    # A client-credentials token acts as the app itself, so issues the sync
    # creates are not attributed to a person (nor synced to GitHub under one).
    body = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": os.environ["LINEAR_CLIENT_SECRET"],
            "scope": "read,write",
        }
    ).encode()
    with urllib.request.urlopen(LINEAR_TOKEN_URL, data=body) as response:
        return f"Bearer {json.load(response)['access_token']}"


class _Client:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.repo = os.environ["GITHUB_REPOSITORY"]
        self._linear_auth = _linear_auth()
        self._github_token = os.environ["GITHUB_TOKEN"]
        self.triage_team = os.environ.get("LINEAR_TRIAGE_TEAM") or None
        self._milestones: dict[str, int] | None = None

    def linear(self, query: str, **variables: Any) -> dict[str, Any]:
        body = json.dumps({"query": query, "variables": variables}).encode()
        request = urllib.request.Request(
            LINEAR_URL,
            data=body,
            headers={
                "Authorization": self._linear_auth,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            # Linear reports some GraphQL errors (rate limits, bad input) as 4xx
            # with the usual error body.
            try:
                payload = json.load(error)
            except ValueError:
                raise error from None
        if payload.get("errors"):
            raise _LinearError(payload["errors"])
        return payload["data"]

    def github(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        request = urllib.request.Request(
            f"{GITHUB_API}{path}",
            data=json.dumps(data).encode() if data is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self._github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def github_paged(self, path: str) -> list[Any]:
        items: list[Any] = []
        page = 1
        while True:
            sep = "&" if "?" in path else "?"
            batch = self.github("GET", f"{path}{sep}per_page=100&page={page}")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def _load_milestones(self) -> dict[str, int]:
        # state=all: a PR can still reference a release whose milestone was
        # already closed, and reopening a duplicate title would fail.
        return {
            m["title"]: m["number"]
            for m in self.github_paged(f"/repos/{self.repo}/milestones?state=all")
        }

    def milestone_number(self, title: str) -> int:
        if self._milestones is None:
            self._milestones = self._load_milestones()
        if title not in self._milestones:
            _log(f"create milestone {title}")
            if self.dry_run:
                self._milestones[title] = -1
                return -1
            try:
                created = self.github(
                    "POST", f"/repos/{self.repo}/milestones", {"title": title}
                )
                self._milestones[title] = created["number"]
            except urllib.error.HTTPError as error:
                if error.code != 422:
                    raise
                # 422 is either a concurrent run creating it first, or a real
                # validation error; only the former shows up on reload.
                self._milestones = self._load_milestones()
                if title not in self._milestones:
                    raise
        return self._milestones[title]


@dataclass
class _LinkedIssue:
    identifier: str
    release: str | None


@dataclass
class _Attachment:
    id: str
    issue: _LinkedIssue
    link_kind: str | None

    @property
    def inherited(self) -> bool:
        return self.link_kind == INHERITED_LINK_KIND


def _log(message: str) -> None:
    print(message, flush=True)


def _referenced_ids(pr: dict[str, Any]) -> list[str]:
    # Branch and magic-word IDs win, so an ID merely mentioned in the
    # description cannot override the issue the PR is actually linked to.
    branch = [f"{t}-{n}" for t, n in ISSUE_ID_RE.findall(pr["head"]["ref"])]
    text = "\n".join([pr["title"], pr.get("body") or ""])
    linked = branch + MAGIC_ID_RE.findall(text)
    if not linked:
        linked = [f"{t}-{n}" for t, n in ISSUE_ID_RE.findall(text)]
    return list(dict.fromkeys(i.upper() for i in linked))


def _referenced_github_issues(pr: dict[str, Any], repo: str) -> list[int]:
    # Only after a magic word: a bare "#12" may just as well be a PR.
    target = rf"(?:https://github\.com/{re.escape(repo)}/issues/|{re.escape(repo)}#|#)"
    pattern = re.compile(MAGIC_WORDS + target + r"(\d+)\b", re.IGNORECASE)
    text = "\n".join([pr["title"], pr.get("body") or ""])
    return list(dict.fromkeys(int(n) for n in pattern.findall(text)))


def _release_of(milestone: dict[str, Any] | None) -> str | None:
    if not milestone:
        return None
    match = RELEASE_RE.match(milestone["name"])
    return f"v{match.group(1)}" if match else None


def _fetch_issue(client: _Client, identifier: str) -> _LinkedIssue | None:
    # Any error other than "not found" propagates: treating an outage as "no
    # issue" would clear the milestone of every open PR.
    try:
        data = client.linear(
            """query($id: String!) {
              issue(id: $id) { identifier team { key } projectMilestone { name } }
            }""",
            id=identifier,
        )
    except _LinearError as error:
        if not error.not_found:
            raise
        _log(f"  {identifier}: not found")
        return None
    issue = data["issue"]
    # An issue moved to another team still resolves by its old identifier.
    if issue["team"]["key"] != TEAM_KEY:
        _log(f"  {identifier}: moved out of {TEAM_KEY}")
        return None
    return _LinkedIssue(issue["identifier"], _release_of(issue["projectMilestone"]))


def _attachments(client: _Client, pr_url: str) -> list[_Attachment]:
    data = client.linear(
        """query($url: String!) {
          attachmentsForURL(url: $url) {
            nodes {
              id
              metadata
              issue { identifier team { key } projectMilestone { name } }
            }
          }
        }""",
        url=pr_url,
    )
    return [
        _Attachment(
            node["id"],
            _LinkedIssue(
                node["issue"]["identifier"],
                _release_of(node["issue"]["projectMilestone"]),
            ),
            (node["metadata"] or {}).get("linkKind"),
        )
        for node in data["attachmentsForURL"]["nodes"]
        if node["issue"]["team"]["key"] in _linked_teams(client)
    ]


def _linked_teams(client: _Client) -> set[str]:
    # An issue still in triage counts as linked: it just has no release yet.
    return {TEAM_KEY} | ({client.triage_team} if client.triage_team else set())


def _attach_inherited(
    client: _Client,
    pr: dict[str, Any],
    issue: _LinkedIssue,
    attached: list[_Attachment],
) -> None:
    # Persisted so the child keeps the issue after its parent merges. Only
    # inherited attachments are ever on a child here (any other kind wins over
    # the parent), so these came from an earlier parent issue.
    if any(a.issue.identifier == issue.identifier for a in attached):
        return
    for stale in attached:
        _log(f"  detach from {stale.issue.identifier}")
        if not client.dry_run:
            client.linear(
                "mutation($id: String!) { attachmentDelete(id: $id) { success } }",
                id=stale.id,
            )
    _log(f"  attach to inherited {issue.identifier}")
    if not client.dry_run:
        _link_pr(client, issue.identifier, pr["html_url"], INHERITED_LINK_KIND)


def _link_pr(client: _Client, issue_id: str, pr_url: str, kind: str) -> None:
    # A GitHub PR attachment (not a plain link) so Linear's own PR automation
    # tracks review and merge state from here on.
    try:
        data = client.linear(
            """mutation($issueId: String!, $url: String!, $kind: GitLinkKind) {
              attachmentLinkGitHubPR(issueId: $issueId, url: $url, linkKind: $kind) {
                success
              }
            }""",
            issueId=issue_id,
            url=pr_url,
            kind=kind,
        )
    except _LinearError as error:
        # Re-linking is not idempotent: Linear rejects an already attached URL.
        if not error.duplicate_attachment:
            raise
        _log(f"  {pr_url} already attached")
        return
    if not data["attachmentLinkGitHubPR"]["success"]:
        raise RuntimeError(f"Linear did not attach {pr_url} to {issue_id}")


def _triage_description(pr: dict[str, Any]) -> str:
    # Linear stores a bare URL as [url](<url>); written that way up front, the
    # marker below matches what Linear keeps.
    url = pr["html_url"]
    return f"Opened from [{url}](<{url}>) by @{pr['user']['login']}."


def _triage_marker(pr: dict[str, Any]) -> str:
    # The trailing ">) by" keeps pull/1 from matching pull/10.
    return f"(<{pr['html_url']}>) by"


def _has_triage_comment(client: _Client, pr: dict[str, Any]) -> bool:
    comments = client.github_paged(
        f"/repos/{client.repo}/issues/{pr['number']}/comments"
    )
    return any(TRIAGE_COMMENT_MARKER in (c.get("body") or "") for c in comments)


def _record_triage(client: _Client, pr: dict[str, Any], issue: _LinkedIssue) -> None:
    if client.dry_run:
        return
    _link_pr(client, issue.identifier, pr["html_url"], "closes")
    client.github(
        "POST",
        f"/repos/{client.repo}/issues/{pr['number']}/comments",
        {"body": f"{TRIAGE_COMMENT_MARKER}\nTracked in Linear for triage."},
    )


def _create_triage_issue(client: _Client, pr: dict[str, Any]) -> _LinkedIssue | None:
    if not client.triage_team:
        _log("  no LINEAR_TRIAGE_TEAM, no triage issue")
        return None
    # Reaching here with a recorded triage issue means triage moved it to a team
    # the key can't see (or detached it): the PR is handled, don't open another.
    if _has_triage_comment(client, pr):
        _log("  triage issue already opened and no longer visible, skipped")
        return None
    # Creating and attaching are separate mutations. If attaching failed on an
    # earlier run, the issue exists but no attachment points at it, so look it up
    # by its description before creating another.
    existing = client.linear(
        """query($keys: [String!]!, $marker: String!) {
          issues(
            first: 1
            filter: { team: { key: { in: $keys } }, description: { contains: $marker } }
          ) { nodes { identifier projectMilestone { name } } }
        }""",
        keys=sorted(_linked_teams(client)),
        marker=_triage_marker(pr),
    )["issues"]["nodes"]
    if existing:
        issue = _LinkedIssue(
            existing[0]["identifier"], _release_of(existing[0]["projectMilestone"])
        )
        _log(f"  reuse triage issue {issue.identifier} for #{pr['number']}")
        _record_triage(client, pr, issue)
        return issue

    _log(f"  create {client.triage_team} triage issue for #{pr['number']}")
    if client.dry_run:
        return None
    team = client.linear(
        """query($key: String!) {
          teams(filter: { key: { eq: $key } }) {
            nodes {
              id
              states(filter: { type: { eq: "triage" } }) { nodes { id } }
              labels(filter: { name: { eq: "%s" } }) { nodes { id } }
            }
          }
        }"""
        % TRIAGE_LABEL,
        key=client.triage_team,
    )["teams"]["nodes"][0]
    issue_input: dict[str, Any] = {
        "teamId": team["id"],
        "title": pr["title"],
        "description": _triage_description(pr),
    }
    if team["states"]["nodes"]:
        issue_input["stateId"] = team["states"]["nodes"][0]["id"]
    if team["labels"]["nodes"]:
        issue_input["labelIds"] = [team["labels"]["nodes"][0]["id"]]
    created = client.linear(
        "mutation($input: IssueCreateInput!) { issueCreate(input: $input) { issue { id identifier } } }",
        input=issue_input,
    )["issueCreate"]["issue"]
    issue = _LinkedIssue(created["identifier"], None)
    _record_triage(client, pr, issue)
    _log(f"  created {issue.identifier}")
    return issue


def _synced_issue(client: _Client, number: int) -> _LinkedIssue | None:
    # Issue sync attaches the GitHub issue to its Linear twin.
    url = f"https://github.com/{client.repo}/issues/{number}"
    attached = _attachments(client, url)
    if not attached:
        _log(f"  #{number}: not synced to Linear")
        return None
    return attached[0].issue


def _stack_parent(
    client: _Client, pr: dict[str, Any], default_branch: str
) -> dict[str, Any] | None:
    base = pr["base"]["ref"]
    if base == default_branch:
        return None
    owner = client.repo.split("/")[0]
    head = urllib.parse.quote(f"{owner}:{base}", safe=":")
    parents = client.github("GET", f"/repos/{client.repo}/pulls?state=open&head={head}")
    return parents[0] if parents else None


def _is_bot(pr: dict[str, Any]) -> bool:
    return pr["user"]["type"] == "Bot"


def _resolve(
    client: _Client,
    pr: dict[str, Any],
    default_branch: str,
    create: bool,
    depth: int = 0,
) -> tuple[_LinkedIssue | None, bool]:
    """Return the PR's ZEPH issue, and whether it references only PLAT issues."""
    ids = _referenced_ids(pr)
    zeph = [i for i in ids if i.startswith(f"{TEAM_KEY}-")]
    github_issues = _referenced_github_issues(pr, client.repo)
    # Lazy, so lookups stop at the first issue with a release.
    candidates = itertools.chain(
        (_fetch_issue(client, i) for i in zeph),
        (_synced_issue(client, n) for n in github_issues),
    )
    fallback: _LinkedIssue | None = None
    for issue in candidates:
        if issue and issue.release:
            return issue, False
        fallback = fallback or issue
    if fallback:
        return fallback, False
    # Checked before attachments: a linked PLAT issue fails even if the PR is
    # also attached to a ZEPH issue, since the reference itself is the leak.
    if ids and not zeph and not github_issues:
        return None, True

    # An attachment someone made (in Linear, or a triage issue) is as explicit
    # as an ID in the PR and wins over the parent.
    attached = _attachments(client, pr["html_url"])
    own = [a for a in attached if not a.inherited]
    if own:
        return own[0].issue, False

    # Stacked PRs inherit their parent's issue, so a stack without IDs yields one
    # triage issue (from the PR targeting the default branch), not one per PR.
    # The live parent is checked before inherited attachments so a parent
    # switching issues carries its children along. Only the issue is inherited:
    # a parent's PLAT reference is reported on the parent's own run.
    parent = _stack_parent(client, pr, default_branch)
    if parent and depth < 20:
        issue, _ = _resolve(client, parent, default_branch, create, depth + 1)
        if issue:
            if depth == 0:
                _attach_inherited(client, pr, issue, attached)
            return issue, False

    if attached:
        return attached[0].issue, False

    # An unresolvable ZEPH ID (typo, issue moved away) is a broken link, not a
    # missing one, so it gets no triage issue. Neither does a stacked child: its
    # parent's own run opens the one for the stack.
    if create and not ids and not parent and not _is_bot(pr) and depth == 0:
        return _create_triage_issue(client, pr), False
    return None, False


def _sync_milestone(
    client: _Client, pr: dict[str, Any], issue: _LinkedIssue | None
) -> None:
    current = (pr.get("milestone") or {}).get("title")
    wanted = issue.release if issue else None
    if wanted == current:
        return
    if current is not None and not MANAGED_MILESTONE_RE.match(current):
        return
    if wanted is None:
        _log(f"  clear milestone {current}")
        number = None
    else:
        _log(f"  milestone {current or '-'} -> {wanted}")
        number = client.milestone_number(wanted)
    if not client.dry_run:
        client.github(
            "PATCH",
            f"/repos/{client.repo}/issues/{pr['number']}",
            {"milestone": number},
        )


def _handle(
    client: _Client, pr: dict[str, Any], default_branch: str, create: bool
) -> bool:
    """Process one PR; return False if it references only PLAT issues."""
    _log(f"#{pr['number']} {pr['title']}")
    # Dependabot runs get no Actions secrets, so an ID-free bot PR must not
    # reach Linear at all.
    if _is_bot(pr) and not _referenced_ids(pr):
        _log("  bot PR without issue ID, skipped")
        return True
    issue, plat_only = _resolve(client, pr, default_branch, create)
    if plat_only:
        _log(
            f"::error::#{pr['number']} references a PLAT issue only. Zephon PRs must "
            f"reference a {TEAM_KEY} issue; create or move the issue to {TEAM_KEY}."
        )
        return False
    _log(f"  linked to {issue.identifier if issue else '-'}")
    _sync_milestone(client, pr, issue)
    return True


def main() -> int:
    """Run one sync pass; return the process exit code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pr", "reconcile"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    client = _Client(args.dry_run)
    default_branch = client.github("GET", f"/repos/{client.repo}")["default_branch"]

    if args.mode == "pr":
        with open(os.environ["GITHUB_EVENT_PATH"]) as f:
            event = json.load(f)
        # Re-fetch: the event payload can predate an edit made seconds earlier.
        number = event["pull_request"]["number"]
        pr = client.github("GET", f"/repos/{client.repo}/pulls/{number}")
        return 0 if _handle(client, pr, default_branch, create=True) else 1

    # Reconcile never creates triage issues or fails: it only repairs milestones,
    # and a PLAT reference is reported on that PR's own run.
    for pr in client.github_paged(f"/repos/{client.repo}/pulls?state=open"):
        try:
            _handle(client, pr, default_branch, create=False)
        # One PR's failure (network, unexpected payload) must not stop the pass.
        except Exception as error:
            _log(f"::warning::#{pr['number']}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
