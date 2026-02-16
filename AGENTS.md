# AGENT.md - Development Guide

⚠️ **CRITICAL: This project uses Beads for ALL task tracking. NEVER use built-in TaskCreate/TaskUpdate tools.**

Follow these instructions precisely for all sessions.

## Python
- Use uv for package management and running scripts
- Sphinx-style docstrings
- Include type hints where feasible
- prefer i_noun over noun_idx

## Coding style
- Reuse existing functions rather than duplicating
- Keep related functionality together

## Text
- don't use emoticons, instead use unicode line icons

## Comments:
- Avoid obvious comments
- Only comment non-self-evident logic

## Philosophy:
- Keep it simple and focused
- Don't over-engineer or add unnecessary features
- Minimum complexity needed for the current task

## Beads Workflow (MANDATORY)

This project uses [Beads](https://github.com/steveyegge/beads) for issue tracking. **Use `bd` commands instead of markdown TODOs.**

### Session Start
1. Run `bd ready` to list unblocked issues
2. Select highest-priority matching issue
3. Claim it: `bd update <id> --status=in_progress`

### During Work
- **ALWAYS use Beads for tasks** - NEVER use TaskCreate/TaskUpdate tools
- Create issues: `bd create --title="..." --type=task --priority=2`
- Update progress: `bd update <id> --append-notes="Progress..."` ⚠️ **NOT --note**
- Add dependencies: `bd dep add <child> <blocks>`
- Types: task, bug, feature, epic, question, docs
- Priorities: P0 (critical) → P4 (backlog)

**Common mistakes to avoid:**
- ❌ `--note` (doesn't exist) → ✅ `--append-notes`

### Breaking Down Epics into Subtasks (IMPORTANT)

When implementing epics or large features:

1. **Create subtasks** using Beads parent-child structure:
   ```bash
   bd create --title="Subtask name" --type=task --priority=1 --parent=<epic-id>
   bd dep add <child-id> <dependency-id>  # Set dependencies between subtasks
   ```

2. **Identify parallelizable work**: Tasks are independent if they:
   - Modify different files (no merge conflicts)
   - Have no data dependencies (one doesn't need other's output)
   - Can be implemented without seeing each other's code

3. **Always use worktrees for parallel subtasks** - this is MANDATORY for parallel work

### Parallel Agent Workflow (Advanced)

For **independent tasks** that can be worked in parallel, use git worktrees + sub-agents:

**When to use:**
- Subtasks of an epic that are truly independent
- Multiple independent refactorings in different files
- Separate feature implementations
- Tasks with no dependencies

**Setup:**
```bash
# 1. Check for uncommitted changes and handle them
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Uncommitted changes detected - commit or stash before parallel work"
    git status
    # Either commit the changes or stash them:
    # Option A: git add -A && git commit -m "WIP: commit before parallel work"
    # Option B: git stash push -m "Stashing before parallel work"
    exit 1
fi

# 2. Create beads tasks
bd create --title="Task 1" --type=task --priority=2
bd create --title="Task 2" --type=task --priority=2
bd create --title="Task 3" --type=task --priority=2

# 3. Create worktrees with branches (from main conversation)
REPO=$(basename $(pwd))
git worktree add ../${REPO}-task1 -b task/task1-description
git worktree add ../${REPO}-task2 -b task/task2-description
git worktree add ../${REPO}-task3 -b task/task3-description
```

**Launch agents (single message, multiple Task tool calls):**
```python
Task(subagent_type="general-purpose",
     description="Work on task 1",
     prompt="""
     cd ../${REPO}-task1
     bd update <task1-id> --status=in_progress

     [Implement task 1]

     git add -A && git commit -m "Close <task1-id>: description"
     bd close <task1-id> --reason="Completed"
     git push -u origin task/task1-description
     """)

Task(subagent_type="general-purpose", ...)  # Similar for task 2
Task(subagent_type="general-purpose", ...)  # Similar for task 3
```

**Agent responsibilities:**
1. `cd` into their worktree
2. Claim task with `bd update --status=in_progress`
3. Implement changes
4. Commit with "Close <id>: ..." message
5. Close task with `bd close <id>`
6. Push branch to remote

**Main conversation cleanup:**
```bash
# After agents complete:
# 0. Handle any uncommitted changes BEFORE merging
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Stashing uncommitted changes before merge"
    git stash push -m "Stashing before merging parallel work"
fi

# 1. Review and merge branches
git checkout main
git merge task/task1-description --no-edit
git merge task/task2-description --no-edit
git merge task/task3-description --no-edit

# 2. Apply stashed changes if any (will prompt if conflicts)
if git stash list | grep -q "Stashing before merging parallel work"; then
    echo "Applying stashed changes..."
    git stash pop
fi

# 3. Push merged work
git push

# 4. Clean up worktrees
git worktree remove ../${REPO}-task1
git worktree remove ../${REPO}-task2
git worktree remove ../${REPO}-task3

# 5. Delete branches
git branch -d task/task1-description task/task2-description task/task3-description
git push origin --delete task/task1-description task/task2-description task/task3-description

# 6. Sync beads
bd sync
```

**Critical rules:**
- ❌ NEVER use for tasks that modify the same files
- ❌ NEVER use for tasks with dependencies
- ✅ ONLY use for truly independent work
- ✅ Each agent works in isolated worktree (no conflicts)
- ✅ Main conversation coordinates and merges results
- ✅ When asked to implement an epic/phase "with subagents", ALWAYS:
  1. Break into subtasks with Beads
  2. Create worktrees for independent subtasks
  3. Launch agents in parallel where possible
  4. Merge results in main conversation

### After Each Task (MANDATORY)
1. Close the issue: `bd close <id> --reason="..."`
2. Stage and commit: `git add -A && git commit -m "Close <id>: description"`
3. If there is a relevant spec/migration md file, update that too
4. Only then move to the next task

This ensures atomic commits per task and clean git history.

### Session End
1. Close completed issues: `bd close <id>`
2. Create follow-up issues if needed
3. Run `bd sync` to commit/push
4. Leave git clean


## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
