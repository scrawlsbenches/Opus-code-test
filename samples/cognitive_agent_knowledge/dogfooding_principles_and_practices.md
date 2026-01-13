# Dogfooding Principles and Practices

*A teaching document for AI agents on using your own tools to build and improve them.*

---

## What Is Dogfooding?

**Dogfooding** means using the product you build to do the work of building it. The term comes from "eating your own dog food" - if you make dog food, you should be willing to feed it to your own dog.

In software, this means:
- Using your search system to find code in your search system
- Using your task tracker to track work on your task tracker
- Using your text processor to process your text processor's documentation

This is not vanity or recursion for its own sake. It is a **strategic practice** that creates a tight feedback loop between creator and creation.

---

## Why Dogfooding Matters

### The Feedback Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     THE DOGFOODING FEEDBACK LOOP                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌───────────┐        ┌───────────┐        ┌───────────┐              │
│   │   BUILD   │───────▶│    USE    │───────▶│   LEARN   │              │
│   │  feature  │        │  feature  │        │   issues  │              │
│   └───────────┘        └───────────┘        └─────┬─────┘              │
│         ▲                                         │                     │
│         │                                         │                     │
│         └─────────────────────────────────────────┘                     │
│                                                                          │
│   Without dogfooding: Build → Ship → Hope users report bugs             │
│   With dogfooding:    Build → Use immediately → Fix before shipping     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Three Core Benefits

**1. You Find Bugs Before Users Do**

When you use your own tool daily, you encounter bugs in realistic conditions. You discover edge cases that unit tests miss because you use the tool under real-world pressure.

Example from this codebase: The cognitive agent was unable to answer "What is TextToAtomsBridge?" because it had never been trained on its own source code. This was only discovered when an AI agent actually tried to use the system for context recovery. A unit test would have verified that the `ask()` function works - but wouldn't reveal that the model lacked useful knowledge.

**2. You Understand User Experience**

Documentation looks clear when you write it. Interfaces seem intuitive when you design them. But when you actually use them - especially after time has passed - you feel the friction. You notice when error messages are cryptic, when workflows are clunky, when features are hard to discover.

This is **embodied understanding**. It's the difference between saying "this should work" and knowing "this is how it works when I'm tired and frustrated."

**3. You Build Empathy for Your Users**

When your own productivity depends on the quality of your tool, you feel the cost of every bug and limitation. This aligns your incentives with your users' needs. You stop thinking "good enough" and start thinking "is this actually good?"

---

## How to Identify Dogfooding Opportunities

Ask these questions about any system you build:

### The Self-Reference Test

Can this system operate on itself?

| System | Self-Reference Question |
|--------|------------------------|
| Search engine | Can it search its own code? |
| Task tracker | Can it track its own development? |
| Documentation generator | Can it document itself? |
| Text processor | Can it process its own docs? |
| Learning system | Can it learn about itself? |

If the answer is "yes" - you should be dogfooding.

### The Daily Use Test

Do you use this system every day? Could you?

If you build a note-taking app but use a different app for your own notes, you're missing feedback. If you build a search system but use grep for your own searches, you're missing friction signals.

### The Canary Test

Would problems in this system hurt your own work?

A system you depend on is a system you'll notice problems in. Make yourself a customer of your own product.

---

## Dogfooding in This Codebase

This codebase practices dogfooding at multiple levels. Study these examples:

### Example 1: CognitiveAgent Learning About Itself

The CognitiveAgent is trained on the `cortical/` source code that implements it:

```bash
# Train the cognitive agent on its own implementation
python -m cortical.cognitive train cortical/ --pattern "*.py"
```

**Why this matters:**
- When you ask "What is CognitiveGraph?", the agent can answer because it has processed `cortical/cognitive/graph.py`
- When you ask "How does training work?", it has seen `cortical/cognitive/training.py`
- The agent's ability to help you understand the codebase IS the test of whether it works

**The bug this caught:** The model was trained only on `samples/` documents, not on `cortical/`. It could answer questions about sample content but couldn't explain its own components. This failure was only discovered when an agent tried to use it for real work.

**The lesson:** A system that helps others understand code should understand its own code first. If it can't explain itself, it can't be trusted to explain anything.

### Example 2: GoT Tracking Its Own Development

The Graph of Thought (GoT) system tracks tasks, decisions, and knowledge transfers. And it tracks the development of GoT itself:

```bash
# Tasks for GoT development are tracked in GoT
python -m cortical.got task list --status in_progress

# Decisions about GoT architecture are logged in GoT
python -m cortical.got decision list

# Knowledge transfers about GoT are stored in GoT
python -m cortical.got kt list
```

**Why this matters:**
- If the task system has bugs, you discover them while using it
- If the query language is awkward, you feel it every time you check status
- If handoffs lose context, you experience that loss personally

**The feedback this creates:** Improvements to GoT immediately improve GoT development. Pain points are felt immediately and fixed quickly.

### Example 3: Codebase Search Using Its Own Algorithms

The Cortical Text Processor provides semantic search. It uses that search to find code within itself:

```bash
# Using the system to search for how the system works
python scripts/search_codebase.py "PageRank algorithm"
```

**Why this matters:**
- If search returns irrelevant results, you know immediately
- If query expansion goes wrong, you experience it
- If indexing misses files, you can't find your own code

**The bug this caught:** Document-type boosting wasn't being applied to passage-level search. This meant queries like "what is a minicolumn" returned code instead of documentation. Only discovered by actually searching the codebase for concepts.

---

## The Risks of Dogfooding

Dogfooding is powerful but not without risks. Be aware of these pitfalls:

### Risk 1: Blind Spots

You know your system too well. You work around bugs without noticing them. You understand cryptic error messages because you wrote them. You navigate confusing UI because you remember what each button does.

**Symptom:** Users report obvious problems you never saw.

**Mitigation:**
- Rotate dogfooders - fresh eyes see fresh problems
- Document your workarounds - if you work around something, it's a bug
- Watch others use the system - observe without helping

### Risk 2: Echo Chambers

If you only build features you need, you ignore features others need. Your use cases become the only use cases. The system becomes optimized for one user (you) instead of many.

**Symptom:** Users request "obvious" features that never occurred to you.

**Mitigation:**
- Collect feedback from multiple sources, not just yourself
- Prioritize features by user count, not personal need
- Ask: "Who else would use this? How?"

### Risk 3: Over-Engineering for Self

You might gold-plate features you use while neglecting features others need. Your workflow gets optimized while other workflows remain broken.

**Symptom:** "Power user" features are polished; "basic" features are rough.

**Mitigation:**
- Track feature usage metrics, not just personal preference
- Ensure core paths work well before optimizing advanced paths
- Get feedback from new users, not just experts

### Risk 4: Rationalization

When you depend on your own tool, you have incentive to believe it's good. You might rationalize bugs as "features" or dismiss complaints as "user error."

**Symptom:** You defend the system instead of improving it.

**Mitigation:**
- Take complaints seriously, even when they seem wrong
- Assume the user is right until proven otherwise
- Record all issues, even ones you disagree with

---

## Best Practices for Effective Dogfooding

### Practice 1: Use It Daily

Sporadic usage doesn't build intuition. Daily usage creates embodied knowledge of what works and what doesn't.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PRINCIPLE: If you don't use it daily, you don't really know it.        │
└─────────────────────────────────────────────────────────────────────────┘
```

Schedule dogfooding into your workflow:
- Start each session with a search query
- Track all tasks in your task system
- Document findings in your documentation system

### Practice 2: Document Problems Immediately

When you hit friction, write it down. Don't say "I'll fix this later" - you'll forget. Don't say "this isn't that bad" - you've adapted.

```bash
# In this codebase:
python -m cortical.got task create "Search returns confusing results for X" \
    --priority medium \
    --category bugfix
```

The act of documenting forces clarity. "It's weird" becomes "When I search for X, I expect Y but get Z because of A."

### Practice 3: Compare Against Alternatives

Use both your tool and alternatives. Notice differences. Ask why.

- Search with your system AND with grep. Different results? Why?
- Track tasks in your system AND in your head. Different outcomes? Why?
- Write docs with your generator AND manually. Different quality? Why?

The comparison reveals where your tool excels and where it fails.

### Practice 4: Fresh Eyes Regularly

After you've used the system for weeks, bring in someone new. Watch them struggle. Resist the urge to help. Their confusion reveals your blind spots.

In an AI agent context: start a new session without context. Try to accomplish a task. Notice what's hard without prior knowledge.

### Practice 5: Measure and Track

Don't rely on feelings alone. Track metrics:

| Metric | What It Reveals |
|--------|-----------------|
| Time to complete task | Is the workflow efficient? |
| Number of workarounds | Are there hidden bugs? |
| Retry rate | Are operations reliable? |
| Documentation lookups | Is the interface intuitive? |

### Practice 6: Act on Feedback

Dogfooding is useless if you don't act on what you learn. Create a pipeline from discovery to fix:

```
Discover problem → Document problem → Prioritize fix → Implement fix → Verify fix
```

In this codebase, that looks like:

```bash
# 1. Discover: "Search doesn't find new files"
# 2. Document:
python -m cortical.got task create "Search index doesn't include new files" --priority high

# 3. Prioritize: Review task list, assign priority
python -m cortical.got task list --status pending

# 4. Implement: Fix the bug

# 5. Verify: Use search to confirm fix
python scripts/search_codebase.py "new file content"
```

---

## A Checklist for Dogfooding Sessions

Use this checklist when dogfooding:

```
□ BEFORE STARTING
  □ Clear your mental model - approach like a new user
  □ Have a real task, not a fake test scenario
  □ Prepare to document issues as you go

□ DURING USE
  □ Note every friction point, even small ones
  □ Note every positive surprise
  □ Don't work around bugs - document them
  □ Compare results to expectations

□ AFTER FINISHING
  □ Review notes and create actionable tasks
  □ Prioritize issues by impact
  □ Share findings with others
  □ Schedule follow-up to verify fixes
```

---

## The Deeper Lesson

Dogfooding is more than a testing technique. It is a **philosophy of accountability**.

When you use your own tools, you can't hide from their flaws. Every bug costs you time. Every limitation frustrates you. Every improvement helps you.

This creates alignment between creator and creation. You stop building what you think users need and start building what you actually need - and because you're a sophisticated user, what you need often matches what others need.

The Cortical Text Processor indexes its own source code not because it's clever recursion, but because:

1. **It proves the system works** - If it can't understand itself, how can we trust it to understand anything?
2. **It improves the system** - Every bug found is a bug fixed
3. **It builds knowledge** - The system literally learns about itself
4. **It demonstrates value** - If we won't use it, why should anyone?

---

## Questions for Reflection

When approaching any system, ask:

1. **Can I use this to build this?** If yes, you should.
2. **Am I currently using alternatives?** If yes, why? What does my tool lack?
3. **What would I need to use this daily?** Build that.
4. **What problems have I normalized?** Document them.
5. **What would a new user struggle with?** Fix that.

---

## Summary

| Aspect | Key Point |
|--------|-----------|
| Definition | Using your own product to build and improve it |
| Core benefit | Tight feedback loop between creation and usage |
| Main risks | Blind spots, echo chambers, rationalization |
| Best practice | Use daily, document immediately, act on findings |
| Deep lesson | Accountability through personal stake |

**Remember:** The best way to ensure quality is to make yourself depend on that quality. If your productivity suffers when your tool is broken, you will fix it. If it doesn't, you won't.

---

*This document teaches the cognitive agent about dogfooding so it can recognize opportunities, understand benefits and risks, and apply best practices in its own work.*
