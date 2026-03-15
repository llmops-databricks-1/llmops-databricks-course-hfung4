---
name: pr-summary
description: Summarize changes in a pull request including key modifications, impact, and recommendations
context: fork
agent: Explore
allowed-tools: Bash(gh *)
---

## Pull request context
- PR diff: !`gh pr diff`
- PR comments: !`gh pr view --comments`
- Changed files: !`gh pr diff --name-only`

## Your task
Summarize this pull request in a clear, concise way. Your summary should include:

1. **Overall purpose:** What is the PR trying to achieve?
2. **Key changes:** List the main files or modules changed and describe the type of changes (e.g., bug fix, feature addition, refactor).
3. **Impact assessment:** Identify potential side effects, risks, or areas that need attention.
4. **Comment highlights:** Include any important discussion points from PR comments.
5. **Suggestions / Recommendations:** If applicable, mention improvements, refactoring, or follow-up actions.

**Formatting guidelines:**
- Keep the summary **under 200 words** if possible.
- Use bullet points for clarity.
- Reference changed files by name.
- If the PR affects multiple modules, group changes logically.

**Optional:** You can include placeholders for reviewers or stakeholders if needed:
- `$ARGUMENTS` → any user-provided context or emphasis

---

## Example Invocation
