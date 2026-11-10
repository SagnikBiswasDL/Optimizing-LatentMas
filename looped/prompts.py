"""Role prompts for one frozen Ouro backbone.

Every prompt starts with the same ``Question: ...`` block and puts role
instructions after it, so the question tokens form a shared causal prefix across
roles (needed for the cache-reuse design). A causal prefix cannot see the role
text that follows it, so identical question representations across roles say
nothing about whether early loops are "generic".
"""
from __future__ import annotations

from typing import Dict, List

BOX = "put your final answer within \\boxed{}"


def _q(question: str) -> str:
    return f"Question: {question}\n\n"


def msgs(text: str) -> List[Dict]:
    return [{"role": "user", "content": text}]


def solver(question: str) -> str:
    return _q(question) + f"Please reason step by step, and {BOX}."


def planner(question: str) -> str:
    return _q(question) + (
        "You are the Planner. Write a short numbered plan (at most 6 steps) that says "
        "how to solve the question. Do not carry out the arithmetic and do not state the final answer."
    )


def plan_critic(question: str, plan: str) -> str:
    return _q(question) + (
        f"You are the Critic. The Planner proposed this plan:\n{plan}\n\n"
        "Check each step of the plan against the question. Point out any step that is wrong, "
        "missing, or misreads the question, and say how to fix it. If the plan is sound, say so. "
        "Do not solve the problem."
    )


def final_solver(question: str, plan: str, critique: str) -> str:
    return _q(question) + (
        f"Plan from the Planner:\n{plan}\n\nFeedback from the Critic:\n{critique}\n\n"
        f"You are the Final Solver. Use the plan and the feedback where they are correct, "
        f"solve the question, and {BOX}."
    )


def solution_critic(question: str, solution: str) -> str:
    return _q(question) + (
        f"You are the Critic. Here is a candidate solution:\n{solution}\n\n"
        "Check every step. Reply in exactly this format:\n"
        "Verdict: CORRECT or INCORRECT\n"
        "First error: Step <n> (or none)\n"
        "Corrected answer: \\boxed{<number>}"
    )


def solver_with_candidate(question: str, solution: str) -> str:
    """Generic-solver control for the critic pilot: same inputs, no critic role."""
    return _q(question) + f"Here is a candidate solution:\n{solution}\n\nPlease reason step by step, and {BOX}."


def self_revision(question: str, previous: str) -> str:
    return _q(question) + (
        f"Here is your previous solution:\n{previous}\n\n"
        f"Review it carefully for mistakes, fix any you find, and {BOX}."
    )
