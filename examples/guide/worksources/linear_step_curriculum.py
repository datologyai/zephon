"""Ramp the mixture linearly over optimizer steps."""

from zephon.work import PhaseLinearSchedule, StepMixturePlan

# Code ramps from 20% to 60% over the first 8,192 steps, then holds.
plan = StepMixturePlan(
    schedules={
        "code": PhaseLinearSchedule([(0, 0.2), (8_192, 0.6)]),
        "web": PhaseLinearSchedule([(0, 0.8), (8_192, 0.4)]),
    },
    total_steps=80_000,
)
