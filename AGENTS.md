<!-- stratograph:begin (managed section — edits are overwritten) -->
# Code graph (stratograph)

This repo is indexed by [stratograph](https://github.com/sunsided/stratograph).
For code understanding and change-impact analysis, query the graph via the
stratograph MCP server (`stratograph mcp`) or CLI rather than `ls`/`grep`:

- find code: `search`, `definition`, `callers`, `callees`, `neighbors`
- API flow: `endpoints`, `clients`, `who_calls`, `api_impact`
- **blast radius before a change**: `impact <symbol>` (add `--downstream` to
  reach other indexed repos that depend on this one)

See `.claude/skills/stratograph/SKILL.md` for details.
<!-- stratograph:end -->
