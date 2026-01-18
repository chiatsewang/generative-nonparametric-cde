Run `git diff --cached` to review the staged changes.

If no files are staged, inform the user and stop.

Generate a commit message following this format:

prefix: summary

Short introduction.

- Change description
- Change description
- Change description

Rules:

- Choose appropriate prefix: feat:, fix:, refactor:, docs:, test:, chore:, init:
- Summary should be 50 characters or less
- Include bullet points for all significant changes
- Use imperative mood ("add" not "added")
- Output only the commit message as plain text (no code blocks, backticks, or markdown)
- No nested bullets or sections

Display the generated message only.
