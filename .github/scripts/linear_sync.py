"""Keep GitHub PR milestones in step with their ZEPH issues in Linear.

Linear is the source of truth for release planning. A PR inherits the release of
the ZEPH issue it references: a Linear milestone named ``vX.Y ...`` or
``vX.Y.Z ...`` becomes the GitHub milestone ``vX.Y`` or ``vX.Y.Z``. Only that
version string is ever written to GitHub.

Modes:
  pr         Handle the PR in ``$GITHUB_EVENT_PATH`` (pull_request_target).
  reconcile  Re-check every open PR (nightly), catching issues that moved
             between releases or left triage.

Pass ``--dry-run`` to log intended changes without writing anything.
Requires ``LINEAR_API_KEY``, ``GITHUB_TOKEN`` and ``GITHUB_REPOSITORY``.
"""

from __future__ import annotations

import argparse
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
GITHUB_API = "https://api.github.com"
TEAM_KEY = "ZEPH"
TRIAGE_LABEL = "from-github-pr"

ISSUE_ID_RE = re.compile(r"\b(ZEPH|PLAT)-(\d+)\b", re.IGNORECASE)
# Linear's magic words; an ID after one of these is a link, not a mention.
MAGIC_ID_RE = re.compile(
    r"\b(?:close[sd]?|closing|fix(?:e[sd])?|fixing|resolve[sd]?|resolving"
    r"|complete[sd]?|completing|refs?|references|part of|related to"
    r"|contributes to|towards?)\s*:?\s+((?:ZEPH|PLAT)-\d+)\b",
    re.IGNORECASE,
)
RELEASE_RE = re.compile(r"^v(\d+\.\d+(?:\.\d+)?)\b")
# Only milestones this script could have set are ever replaced or cleared, so a
# milestone a maintainer set by hand for another reason is left alone.
MANAGED_MILESTONE_RE = re.compile(r"^v\d+\.\d+(?:\.\d+)?$")
# Marks the PR comment recording a triage issue. The key may not see the issue
# once triage moves it to another team, so GitHub keeps the record.
TRIAGE_COMMENT_MARKER = "<!-- linear-sync:triage -->"


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


class _Client:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.repo = os.environ["GITHUB_REPOSITORY"]
        self._linear_key = os.environ["LINEAR_API_KEY"]
        self._github_token = os.environ["GITHUB_TOKEN"]
        self._milestones: dict[str, int] | None = None

    def linear(self, query: str, **variables: Any) -> dict[str, Any]:
        body = json.dumps({"query": query, "variables": variables}).encode()
        request = urllib.request.Request(
            LINEAR_URL,
            data=body,
            headers={
                "Authorization": self._linear_key,
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


def _attachments(client: _Client, pr_url: str) -> list[tuple[str, _LinkedIssue]]:
    """Return ``(attachment id, issue)`` for each ZEPH issue the PR is attached to."""
    data = client.linear(
        """query($url: String!) {
          attachmentsForURL(url: $url) {
            nodes { id issue { identifier team { key } projectMilestone { name } } }
          }
        }""",
        url=pr_url,
    )
    return [
        (
            node["id"],
            _LinkedIssue(
                node["issue"]["identifier"],
                _release_of(node["issue"]["projectMilestone"]),
            ),
        )
        for node in data["attachmentsForURL"]["nodes"]
        if node["issue"]["team"]["key"] == TEAM_KEY
    ]


def _attach_inherited(client: _Client, pr: dict[str, Any], issue: _LinkedIssue) -> None:
    # Persisted so the child keeps the issue after its parent merges. Any other
    # ZEPH attachment came from an earlier parent issue; the live parent wins.
    attached = _attachments(client, pr["html_url"])
    if any(a.identifier == issue.identifier for _, a in attached):
        return
    for attachment_id, stale in attached:
        _log(f"  detach from {stale.identifier}")
        if not client.dry_run:
            client.linear(
                "mutation($id: String!) { attachmentDelete(id: $id) { success } }",
                id=attachment_id,
            )
    _log(f"  attach to inherited {issue.identifier}")
    if not client.dry_run:
        _link_pr(client, issue.identifier, pr["html_url"], "contributes")


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


def _triage_marker(pr: dict[str, Any]) -> str:
    # The trailing " by" keeps pull/1 from matching pull/10.
    return f"Opened from {pr['html_url']} by"


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
        {"body": f"{TRIAGE_COMMENT_MARKER}\nTracked in Linear as {issue.identifier}."},
    )


def _create_triage_issue(client: _Client, pr: dict[str, Any]) -> _LinkedIssue | None:
    # Reaching here with a recorded triage issue means triage moved it out of
    # ZEPH (or detached it): the PR is handled, so don't open another.
    if _has_triage_comment(client, pr):
        _log("  triage issue already opened and no longer in ZEPH, skipped")
        return None
    # Creating and attaching are separate mutations. If attaching failed on an
    # earlier run, the issue exists but no attachment points at it, so look it up
    # by its description before creating another.
    existing = client.linear(
        """query($key: String!, $marker: String!) {
          issues(
            first: 1
            filter: { team: { key: { eq: $key } }, description: { contains: $marker } }
          ) { nodes { identifier projectMilestone { name } } }
        }""",
        key=TEAM_KEY,
        marker=_triage_marker(pr),
    )["issues"]["nodes"]
    if existing:
        issue = _LinkedIssue(
            existing[0]["identifier"], _release_of(existing[0]["projectMilestone"])
        )
        _log(f"  reuse triage issue {issue.identifier} for #{pr['number']}")
        _record_triage(client, pr, issue)
        return issue

    _log(f"  create {TEAM_KEY} triage issue for #{pr['number']}")
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
        key=TEAM_KEY,
    )["teams"]["nodes"][0]
    issue_input: dict[str, Any] = {
        "teamId": team["id"],
        "title": pr["title"],
        "description": f"{_triage_marker(pr)} @{pr['user']['login']}.",
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
    fallback: _LinkedIssue | None = None
    for identifier in zeph:
        issue = _fetch_issue(client, identifier)
        if issue and issue.release:
            return issue, False
        fallback = fallback or issue
    if fallback:
        return fallback, False
    # Checked before attachments: a linked PLAT issue fails even if the PR is
    # also attached to a ZEPH issue, since the reference itself is the leak.
    if ids and not zeph:
        return None, True

    # Stacked PRs inherit their parent's issue, so a stack without IDs yields one
    # triage issue (from the PR targeting the default branch), not one per PR.
    # The live parent is checked before the PR's own attachment so a parent
    # switching issues carries its children along. Only the issue is inherited:
    # a parent's PLAT reference is reported on the parent's own run.
    parent = _stack_parent(client, pr, default_branch)
    if parent and depth < 20:
        issue, _ = _resolve(client, parent, default_branch, create, depth + 1)
        if issue:
            if depth == 0:
                _attach_inherited(client, pr, issue)
            return issue, False

    attached = _attachments(client, pr["html_url"])
    if attached:
        return attached[0][1], False

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
