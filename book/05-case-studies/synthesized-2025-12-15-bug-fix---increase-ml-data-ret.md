# Case Study: Bug Fix - Increase ML data retention to 2 years for training milestones

*Synthesized from commit history: 2025-12-15*

## The Problem

A new feature was required: Add privacy features to ML data collection. This would require careful implementation and testing.

## The Journey

The development progressed through several stages:

1. **Add privacy features to ML data collection** - Modified 2 files (+628/-1 lines)
2. **Add ML data collection section to README** - Modified 1 files (+89/-0 lines)
3. **Increase ML data retention to 2 years for training milestones** - Modified 2 files (+7/-5 lines)
4. **Add automatic ML data collection on session startup** - Modified 3 files (+95/-22 lines)


## The Solution

Share ML commit data and aggregated patterns in git

The solution involved changes to 474 files, adding 561272 lines and removing 4 lines.


## The Lesson

**Bugs often hide in unexpected places.** Thorough investigation and testing are essential for finding root causes.

## Technical Details

**Files Modified:** 478

- `.claude/settings.local.json`
- `.git-ml/commits/0039ad5b_2025-12-11_24b1b10a.json`
- `.git-ml/commits/00f88d48_2025-12-14_8749d448.json`
- `.git-ml/commits/051d2002_2025-12-13_7896a312.json`
- `.git-ml/commits/051d924c_2025-12-13_bfbb2049.json`
- `.git-ml/commits/059085df_2025-12-10_4adc6156.json`
- `.git-ml/commits/0598bade_2025-12-11_ab9a0c3b.json`
- `.git-ml/commits/061a157b_2025-12-13_af26e95e.json`
- `.git-ml/commits/063c5424_2025-12-10_c2422fa6.json`
- `.git-ml/commits/06567449_2025-12-12_95eabdde.json`

*...and 468 more files*

**Code Changes:** +562091/-32 lines

**Commits:** 5


## Commits in This Story

- `e188508` (2025-12-15): feat: Add privacy features to ML data collection
- `df75750` (2025-12-15): docs: Add ML data collection section to README
- `95e9f06` (2025-12-15): fix: Increase ML data retention to 2 years for training milestones
- `b805e13` (2025-12-15): feat: Add automatic ML data collection on session startup
- `6570973` (2025-12-15): feat: Share ML commit data and aggregated patterns in git

---

*This case study was automatically synthesized from git commit history.*
