"""
MDP environment for Stage 1 of the Multi-Batching Problem.

Key design decisions:
  - Parts routed sequentially (p1 then p2)
  - Action: (from, to, tr, quantity) for the current part
  - qty=0 signals "done routing this part"
  - Step reward: negative delta dispatch cost on the affected arc
  - Terminal reward: large positive for feasibility + cost bonus,
    large negative for violations / unmet demand
"""

import math
from typing import Dict, List, Tuple
from instances import (
    LOCATIONS, PARTS, TRANSPORT, PART_SIZE,
    DEMAND_OFFER, ROUTES, MAX_QUANTITY,
)


def implied_freq(load: Dict, frm: str, to: str, tr: str) -> int:
    capacity = TRANSPORT[tr]["capacity"]
    volume   = sum(load.get((frm, to, tr, part), 0) * PART_SIZE[part] for part in PARTS)
    return math.ceil(volume / capacity) if volume > 0 else 0

def arc_dispatch_cost(load: Dict, frm: str, to: str, tr: str) -> float:
    return implied_freq(load, frm, to, tr) * ROUTES[(frm, to, tr)]["cost"]

def total_network_cost(load: Dict) -> float:
    return sum(arc_dispatch_cost(load, from_location, to_location, transport_resource) for (from_location, to_location, transport_resource) in ROUTES)

def flow_balance(load: Dict, part: str, location: str) -> float:
    """outflow - inflow - net_supply = 0 when conservation holds."""
    inflow  = sum(load.get((from_location, to_location, transport_resource, part), 0) for (from_location, to_location, transport_resource) in ROUTES if to_location == location)
    outflow = sum(load.get((location, to_location, transport_resource, part), 0) for (from_location, to_location, transport_resource) in ROUTES if from_location == location)
    net_supply = DEMAND_OFFER.get((part, location), 0)
    return (outflow - inflow) - net_supply

def total_flow_violation(load: Dict) -> float:
    return sum(abs(flow_balance(load, p, loc)) for p in PARTS for loc in LOCATIONS)

def unmet_demand(load: Dict) -> float:
    total = 0.0
    for p in PARTS:
        for loc in LOCATIONS:
            net = DEMAND_OFFER.get((p, loc), 0)
            if net < 0:
                inflow = sum(load.get((f, loc, tr, p), 0) for (f, t, tr) in ROUTES if t == loc)
                total += max(0, abs(net) - inflow)
    return total


# ── Environment ───────────────────────────────────────────────────────

class MultiBatchingEnv:
    BASELINE_COST = 45.0

    def __init__(
        self,
        lam: float = 10000.0,  
        mu:  float = 1000.0,   
        feasibility_bonus: float = 500.0, 
        max_steps_per_part: int = 30,
    ):
        self.lam               = lam
        self.mu                = mu
        self.feasibility_bonus = feasibility_bonus
        self.max_steps_per_part = max_steps_per_part

        # Actions: (from, to, tr, qty) for qty in 0..MAX_QUANTITY
        self.actions: List[Tuple] = []
        for (frm, to, tr) in sorted(ROUTES.keys()):
            for qty in range(0, MAX_QUANTITY + 1):
                self.actions.append((frm, to, tr, qty))
        self.n_actions = len(self.actions)

        self.reset()

    def reset(self):
        self.load: Dict[Tuple, int] = {}
        self.part_idx: int = 0
        self.steps_this_part: int = 0
        self.done: bool = False
        return self._state_key()

    def _state_key(self) -> Tuple:
        """
        Simplified state: only encode loads for the CURRENT part.
        Since parts are routed sequentially, previous part loads are
        fixed and do not affect current decisions. This keeps the
        state space tractable (~462 reachable states per part).
        """
        if self.part_idx >= len(PARTS):
            return (self.part_idx, ())
        part = PARTS[self.part_idx]
        load_tuple = tuple(
            self.load.get((frm, to, tr, part), 0)
            for (frm, to, tr) in sorted(ROUTES.keys())
        )
        return (self.part_idx, load_tuple)

    def valid_action_indices(self) -> List[int]:
        if self.done or self.part_idx >= len(PARTS):
            return []

        part   = PARTS[self.part_idx]
        supply = DEMAND_OFFER.get((part, "l1"), 0)

        sent_from_source = sum(
            self.load.get(("l1", to, tr, part), 0)
            for (f, to, tr) in ROUTES if f == "l1"
        )
        source_remaining = max(0, supply - sent_from_source)

        valid = []
        for idx, (frm, to, tr, qty) in enumerate(self.actions):
            if qty == 0:
                valid.append(idx)
            elif frm == "l1":
                if qty <= source_remaining:
                    valid.append(idx)
            else:
                # Hub: can only forward what arrived
                arrived = sum(self.load.get(("l1", frm, tr2, part), 0) for tr2 in TRANSPORT)
                forwarded = sum(self.load.get((frm, t, tr2, part), 0)
                                for (f2, t, tr2) in ROUTES if f2 == frm)
                if qty <= max(0, arrived - forwarded):
                    valid.append(idx)
        return valid

    def step(self, action_idx: int) -> Tuple[Tuple, float, bool]:
        assert not self.done
        frm, to, tr, qty = self.actions[action_idx]
        part = PARTS[self.part_idx]

        if qty == 0:
            reward = self._finish_part()
            return self._state_key(), reward, self.done

        cost_before = arc_dispatch_cost(self.load, frm, to, tr)
        key = (frm, to, tr, part)
        self.load[key] = self.load.get(key, 0) + qty
        cost_after = arc_dispatch_cost(self.load, frm, to, tr)

        step_reward = -(cost_after - cost_before)

        self.steps_this_part += 1
        if self.steps_this_part >= self.max_steps_per_part:
            step_reward += self._finish_part()

        return self._state_key(), step_reward, self.done

    def _finish_part(self) -> float:
        self.part_idx       += 1
        self.steps_this_part = 0
        if self.part_idx >= len(PARTS):
            self.done = True
            return self._terminal_reward()
        return 0.0

    def _terminal_reward(self) -> float:
        violations = total_flow_violation(self.load)
        shortage   = unmet_demand(self.load)
        cost       = total_network_cost(self.load)

        if violations == 0 and shortage == 0:
            return self.feasibility_bonus - cost
        else:
            return -(self.lam * violations + self.mu * shortage)

    def summary(self) -> Dict:
        cost       = total_network_cost(self.load)
        violations = total_flow_violation(self.load)
        shortage   = unmet_demand(self.load)
        freqs = {(f, t, tr): implied_freq(self.load, f, t, tr) for (f, t, tr) in ROUTES}
        return {
            "load":           dict(self.load),
            "frequencies":    freqs,
            "total_cost":     cost,
            "flow_violation": violations,
            "unmet_demand":   shortage,
            "feasible":       violations == 0 and shortage == 0,
        }
