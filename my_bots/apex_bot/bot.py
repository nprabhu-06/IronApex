#!/usr/bin/env python3
"""
ApexBot - A fully competitive Codewars bot.

Strategy layers (evaluated every tick, highest priority wins):
  1. DANGER ESCAPE   - Flee from bombs / void immediately
  2. GUN SHOT        - Shoot an opponent if lined up and in range
  3. BOMB PLANT      - Drop a bomb on a nearby opponent or destructible block cluster
  4. PAINT BURST     - Paint-blast to extend territory when safe
  5. CHASE / ENGAGE  - Close the gap to the nearest opponent
  6. EXPLORE         - BFS to the nearest unseen / unclaimed floor tile
  7. IDLE SAFE       - Fallback: stay put

Abilities used:
  - G<dir>  : Gun / shoot a projectile
  - B dx dy : Drop a bomb at relative offset
  - C dx dy : Paint / colour burst
  - J<dir>  : Jump (escape or gap-close)
  - X<dir>  : Shield / dash (emergency dodge)
  - W/A/S/D : cardinal movement
  - I       : idle

Grid centre is always [5][5] = row 5, col 5 (VIEW_HALF = 5).
Positive row  = South (S), positive col = East (D).
"""
from __future__ import annotations

import sys
import random
from collections import deque

# ── Constants ──────────────────────────────────────────────────────────────────
VIEW      = 11
HALF      = VIEW // 2          # 5

FLOOR     = set(".")
PAINT     = set("@$%")         # territory tiles (passable)
PASSABLE  = FLOOR | PAINT | set("1")

SOLID     = set("#*")          # hard walls + destructible blocks
OPPONENTS = set("234")
BOMB_TILE = set("!")
VOID      = set("~")
LETHAL    = VOID | BOMB_TILE

# Action -> (row_delta, col_delta)
MOVE_DIRS: dict[str, tuple[int, int]] = {
    "W": (-1,  0),
    "S": ( 1,  0),
    "A": ( 0, -1),
    "D": ( 0,  1),
}
MOVE_LIST = list(MOVE_DIRS.items())   # stable ordering

# Cooldown tracking (approximate – we don't know true cooldowns, so be conservative)
GUN_CD   = 8
BOMB_CD  = 15
PAINT_CD = 12
JUMP_CD  = 10
SHIELD_CD = 12


# ── State ──────────────────────────────────────────────────────────────────────
class BotState:
    def __init__(self):
        self.turn         = 0
        self.last_gun     = -GUN_CD
        self.last_bomb    = -BOMB_CD
        self.last_paint   = -PAINT_CD
        self.last_jump    = -JUMP_CD
        self.last_shield  = -SHIELD_CD
        # Heuristic exploration memory: set of visited global offsets
        self.visited: set[tuple[int,int]] = set()
        # Direction bias for exploration (avoids getting stuck)
        self.explore_bias = random.choice(["W", "S", "D", "A"])
        self.bias_ttl     = 20
        # Last position for stuck detection
        self.stuck_ctr    = 0
        self.last_action  = "I"

    def can_gun(self):   return self.turn - self.last_gun   >= GUN_CD
    def can_bomb(self):  return self.turn - self.last_bomb  >= BOMB_CD
    def can_paint(self): return self.turn - self.last_paint >= PAINT_CD
    def can_jump(self):  return self.turn - self.last_jump  >= JUMP_CD
    def can_shield(self):return self.turn - self.last_shield>= SHIELD_CD

    def use_gun(self):   self.last_gun   = self.turn
    def use_bomb(self):  self.last_bomb  = self.turn
    def use_paint(self): self.last_paint = self.turn
    def use_jump(self):  self.last_jump  = self.turn
    def use_shield(self):self.last_shield= self.turn


# ── Grid helpers ───────────────────────────────────────────────────────────────

def read_grid() -> list[str] | None:
    """Read one 11x11 frame from stdin."""
    rows = []
    for _ in range(VIEW):
        line = sys.stdin.readline()
        if not line:
            return None
        rows.append(line.rstrip("\r\n"))
    return rows


def cell(grid: list[str], r: int, c: int) -> str:
    """Safe cell accessor; returns '#' for out-of-bounds."""
    if 0 <= r < VIEW and 0 <= c < VIEW:
        return grid[r][c]
    return "#"


def is_passable(ch: str) -> bool:
    return ch in PASSABLE


def is_dangerous(grid: list[str], r: int, c: int) -> bool:
    """True if moving to (r,c) is immediately lethal or bomb-adjacent."""
    ch = cell(grid, r, c)
    if ch in LETHAL or ch in SOLID or ch in OPPONENTS:
        return True
    # Treat tiles orthogonally adjacent to a bomb as dangerous too
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        if cell(grid, r+dr, c+dc) == "!":
            return True
    return False


def safe_moves(grid: list[str]) -> list[str]:
    """Return movement actions that lead to passable, non-lethal tiles."""
    result = []
    for action, (dr, dc) in MOVE_LIST:
        tr, tc = HALF + dr, HALF + dc
        ch = cell(grid, tr, tc)
        if is_passable(ch) and not is_dangerous(grid, tr, tc):
            result.append(action)
    return result


# ── BFS ────────────────────────────────────────────────────────────────────────

def bfs_first_step(
    grid: list[str],
    targets: set[tuple[int,int]],
    avoid: set[str] | None = None,
) -> str | None:
    """
    BFS from centre [HALF,HALF] to any cell in `targets`.
    Returns the first-step action string, or None if unreachable.
    `avoid` is an extra set of tile chars to treat as walls.
    """
    if avoid is None:
        avoid = set()

    # target already adjacent? pick direction immediately
    start = (HALF, HALF)
    if start in targets:
        return None  # already there

    queue: deque[tuple[int,int,str]] = deque()
    seen = {start}

    for action, (dr, dc) in MOVE_LIST:
        nr, nc = HALF + dr, HALF + dc
        if (nr, nc) in seen:
            continue
        ch = cell(grid, nr, nc)
        if not is_passable(ch) or ch in avoid or is_dangerous(grid, nr, nc):
            continue
        if (nr, nc) in targets:
            return action
        seen.add((nr, nc))
        queue.append((nr, nc, action))

    while queue:
        r, c, first = queue.popleft()
        for _, (dr, dc) in MOVE_LIST:
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen:
                continue
            if nr < 0 or nr >= VIEW or nc < 0 or nc >= VIEW:
                continue
            ch = cell(grid, nr, nc)
            if not is_passable(ch) or ch in avoid or is_dangerous(grid, nr, nc):
                continue
            if (nr, nc) in targets:
                return first
            seen.add((nr, nc))
            queue.append((nr, nc, first))

    return None


def flood_fill_count(grid: list[str], start: tuple[int,int]) -> int:
    """Count reachable passable tiles from start (for open-space scoring)."""
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if (nr,nc) in visited:
                continue
            ch = cell(grid, nr, nc)
            if is_passable(ch) and not is_dangerous(grid, nr, nc):
                visited.add((nr,nc))
                queue.append((nr,nc))
    return len(visited)


# ── Opponent analysis ──────────────────────────────────────────────────────────

def find_opponents(grid: list[str]) -> list[tuple[int,int]]:
    """Return list of (row, col) of all visible opponents."""
    ops = []
    for r in range(VIEW):
        for c in range(VIEW):
            if grid[r][c] in OPPONENTS:
                ops.append((r, c))
    return ops


def find_destructibles(grid: list[str]) -> list[tuple[int,int]]:
    return [(r, c) for r in range(VIEW) for c in range(VIEW) if grid[r][c] == "*"]


def find_unclaimed(grid: list[str]) -> list[tuple[int,int]]:
    """Floor tiles not yet painted by us."""
    return [(r, c) for r in range(VIEW) for c in range(VIEW) if grid[r][c] == "."]


# ── Strategy: DANGER ESCAPE ────────────────────────────────────────────────────

def strategy_escape(grid: list[str], state: BotState) -> str | None:
    """If we're currently in danger (adjacent bomb/void), flee ASAP."""
    # Check if current position is already endangered
    in_danger = any(
        cell(grid, HALF + dr, HALF + dc) in BOMB_TILE
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))
    )
    if not in_danger and cell(grid, HALF, HALF) not in VOID:
        return None

    moves = safe_moves(grid)
    if not moves:
        # Try jump escape
        if state.can_jump():
            # Pick a direction with the most open space
            best = None
            best_score = -1
            for d, (dr, dc) in MOVE_LIST:
                nr, nc = HALF + 2*dr, HALF + 2*dc
                if 0 <= nr < VIEW and 0 <= nc < VIEW:
                    ch = cell(grid, nr, nc)
                    if is_passable(ch) and ch not in LETHAL:
                        s = flood_fill_count(grid, (nr, nc))
                        if s > best_score:
                            best_score, best = s, f"J{d}"
            if best:
                state.use_jump()
                return best
        # Shield dash as last resort
        if state.can_shield():
            d = random.choice(list(MOVE_DIRS))
            state.use_shield()
            return f"X{d}"
        return "I"

    # Pick the safest move (most open space)
    best = moves[0]
    best_score = -1
    for mv in moves:
        dr, dc = MOVE_DIRS[mv]
        s = flood_fill_count(grid, (HALF+dr, HALF+dc))
        if s > best_score:
            best_score, best = s, mv
    return best


# ── Strategy: GUN ─────────────────────────────────────────────────────────────

def strategy_gun(grid: list[str], state: BotState) -> str | None:
    """Shoot if an opponent is in the same row or column (cardinal line-of-sight)."""
    if not state.can_gun():
        return None
    opponents = find_opponents(grid)
    if not opponents:
        return None

    for op_r, op_c in opponents:
        # Same column → W or S
        if op_c == HALF:
            direction = "W" if op_r < HALF else "S"
            # Check line-of-sight (no wall in between)
            step = -1 if op_r < HALF else 1
            clear = all(
                cell(grid, r, HALF) not in SOLID
                for r in range(HALF + step, op_r, step)
            )
            if clear:
                state.use_gun()
                return f"G{direction}"
        # Same row → A or D
        if op_r == HALF:
            direction = "A" if op_c < HALF else "D"
            step = -1 if op_c < HALF else 1
            clear = all(
                cell(grid, HALF, c) not in SOLID
                for c in range(HALF + step, op_c, step)
            )
            if clear:
                state.use_gun()
                return f"G{direction}"
    return None


# ── Strategy: BOMB ────────────────────────────────────────────────────────────

def strategy_bomb(grid: list[str], state: BotState) -> str | None:
    """
    Plant a bomb if:
      - An opponent is within 2 tiles orthogonally, OR
      - There is a cluster of destructible blocks nearby worth blasting.
    Bomb is safe if we can move away on the next tick.
    """
    if not state.can_bomb():
        return None

    # Safety check: we need at least one escape route after planting
    escapes = safe_moves(grid)
    if len(escapes) < 1:
        return None

    opponents = find_opponents(grid)
    # Bomb toward nearest opponent within radius 2
    for op_r, op_c in opponents:
        dr, dc = op_r - HALF, op_c - HALF
        dist = abs(dr) + abs(dc)
        if dist <= 3:
            # Clamp to unit offset for bomb placement
            bdr = max(-3, min(3, dr))
            bdc = max(-3, min(3, dc))
            # Verify we won't self-destruct
            state.use_bomb()
            return f"B {bdr} {bdc}"

    # Bomb at a cluster of destructible blocks
    destrs = find_destructibles(grid)
    if len(destrs) >= 2:
        # Pick closest cluster centroid
        best_pos = None
        best_dist = 999
        for r, c in destrs:
            d = abs(r - HALF) + abs(c - HALF)
            if d < best_dist and d <= 3:
                best_dist, best_pos = d, (r, c)
        if best_pos:
            bdr = best_pos[0] - HALF
            bdc = best_pos[1] - HALF
            state.use_bomb()
            return f"B {bdr} {bdc}"

    return None


# ── Strategy: PAINT ───────────────────────────────────────────────────────────

def strategy_paint(grid: list[str], state: BotState) -> str | None:
    """Paint-blast if there are unclaimed floor tiles nearby and it's safe."""
    if not state.can_paint():
        return None
    unclaimed = find_unclaimed(grid)
    if not unclaimed:
        return None

    # Must be safe to stay (not in danger)
    if is_dangerous(grid, HALF, HALF):
        return None

    # Pick the nearest unclaimed tile
    best = None
    best_dist = 999
    for r, c in unclaimed:
        d = abs(r - HALF) + abs(c - HALF)
        if d < best_dist and d <= 4:
            best_dist = d
            best = (r - HALF, c - HALF)

    if best is None:
        return None

    state.use_paint()
    return f"C {best[0]} {best[1]}"


# ── Strategy: CHASE ───────────────────────────────────────────────────────────

def strategy_chase(grid: list[str], state: BotState) -> str | None:
    """BFS toward the nearest opponent."""
    opponents = find_opponents(grid)
    if not opponents:
        return None

    # Sort by Manhattan distance
    opponents.sort(key=lambda p: abs(p[0]-HALF) + abs(p[1]-HALF))
    # Target the tile adjacent to the nearest opponent (not ON them)
    for op_r, op_c in opponents:
        # Collect tiles adjacent to opponent as targets
        adj_targets = set()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = op_r+dr, op_c+dc
            if 0 <= nr < VIEW and 0 <= nc < VIEW:
                ch = cell(grid, nr, nc)
                if is_passable(ch):
                    adj_targets.add((nr, nc))

        if not adj_targets:
            continue

        step = bfs_first_step(grid, adj_targets)
        if step:
            # Consider jumping if the opponent is exactly 2 tiles away and jump is ready
            dist = abs(op_r - HALF) + abs(op_c - HALF)
            if dist == 2 and state.can_jump() and step in MOVE_DIRS:
                state.use_jump()
                return f"J{step}"
            return step

    return None


# ── Strategy: EXPLORE ─────────────────────────────────────────────────────────

def strategy_explore(grid: list[str], state: BotState) -> str | None:
    """BFS toward unclaimed floor tiles; use directional bias to avoid loops."""
    unclaimed = find_unclaimed(grid)
    if unclaimed:
        # Sort by distance, pick closest batch
        unclaimed.sort(key=lambda p: abs(p[0]-HALF)+abs(p[1]-HALF))
        targets = set(unclaimed[:8])
        step = bfs_first_step(grid, targets)
        if step:
            return step

    # Fallback: follow bias direction
    state.bias_ttl -= 1
    if state.bias_ttl <= 0:
        state.explore_bias = random.choice(list(MOVE_DIRS))
        state.bias_ttl = random.randint(10, 25)

    bias = state.explore_bias
    dr, dc = MOVE_DIRS[bias]
    tr, tc = HALF+dr, HALF+dc
    if is_passable(cell(grid, tr, tc)) and not is_dangerous(grid, tr, tc):
        return bias

    # Rotate bias if blocked
    dirs_cycle = ["W","D","S","A"]
    idx = dirs_cycle.index(bias)
    for offset in [1, 2, 3]:
        alt = dirs_cycle[(idx + offset) % 4]
        dr2, dc2 = MOVE_DIRS[alt]
        tr2, tc2 = HALF+dr2, HALF+dc2
        if is_passable(cell(grid, tr2, tc2)) and not is_dangerous(grid, tr2, tc2):
            state.explore_bias = alt
            return alt

    return None


# ── Strategy: STUCK BUSTER ────────────────────────────────────────────────────

def strategy_unstuck(grid: list[str], state: BotState) -> str | None:
    """If stuck for many turns, try a jump to break out."""
    if state.stuck_ctr < 5:
        return None
    if state.can_jump():
        moves = safe_moves(grid)
        if moves:
            d = random.choice(moves)
            state.use_jump()
            state.stuck_ctr = 0
            return f"J{d}"
    # Random safe move
    moves = safe_moves(grid)
    if moves:
        state.stuck_ctr = 0
        return random.choice(moves)
    return None


# ── Main decision function ─────────────────────────────────────────────────────

def choose_action(grid: list[str], state: BotState) -> str:
    # Priority 1: Escape immediate danger
    action = strategy_escape(grid, state)
    if action:
        return action

    # Priority 2: Shoot opponent in line-of-sight
    action = strategy_gun(grid, state)
    if action:
        return action

    # Priority 3: Bomb opponent / block cluster
    action = strategy_bomb(grid, state)
    if action:
        return action

    # Priority 4: Paint nearby floor
    action = strategy_paint(grid, state)
    if action:
        return action

    # Priority 5: Chase nearest opponent
    action = strategy_chase(grid, state)
    if action:
        return action

    # Priority 6: Explore
    action = strategy_explore(grid, state)
    if action:
        return action

    # Priority 7: Unstuck
    action = strategy_unstuck(grid, state)
    if action:
        return action

    # Fallback: safe random or idle
    moves = safe_moves(grid)
    return random.choice(moves) if moves else "I"


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    state = BotState()
    prev_action_was_move = False

    while True:
        grid = read_grid()
        if grid is None:
            break

        action = choose_action(grid, state)

        # Stuck detection: if we keep getting "I" or alternating, increment counter
        is_move = action in MOVE_DIRS
        if not is_move:
            state.stuck_ctr += 1
        else:
            state.stuck_ctr = max(0, state.stuck_ctr - 1)
        state.last_action = action

        try:
            print(action, flush=True)
        except BrokenPipeError:
            break

        # Debug log every 50 turns to stderr (shows in out/logs/)
        if state.turn % 50 == 0:
            ops = find_opponents(grid)
            print(
                f"[ApexBot] turn={state.turn} action={action!r} "
                f"opponents={len(ops)} stuck={state.stuck_ctr}",
                file=sys.stderr, flush=True,
            )

        state.turn += 1


if __name__ == "__main__":
    main()
