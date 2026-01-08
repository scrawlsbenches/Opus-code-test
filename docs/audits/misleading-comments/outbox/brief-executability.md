# Task-001 Executability Assessment

## Can it be executed?
**No** (but very close)

## Missing element
No explicit search command. Agent must infer correct grep/ripgrep syntax for multiple patterns.

## Command that would help most
```bash
rg "(FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:|will be|should be|planned to)" cortical/cdg/ -n --color=never
```

This immediately gives: file:line:content format for all findings without requiring agent interpretation.
