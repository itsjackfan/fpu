import numpy as np

T = 1
ALPHA = 0.4

STATES = [
    "Hangover",
    "Sleep",
    "More Sleep",
    "Visit Lecture",
    "Study",
    "Pass Exam"
]

ACTIONS = [
    "Lazy",
    "Productive",
]

TRANSITIONS = {
    "Hangover": {
        "Lazy": {
            "Sleep": 1.0
        },
        "Productive": {
            "Hangover": 0.7,
            "Visit Lecture": 0.3
        }
    },
    "Sleep": {
        "Lazy": {
            "More Sleep": 1.0
        },
        "Productive": {
            "More Sleep": 0.4,
            "Visit Lecture": 0.6
        }
    },
    "Visit Lecture": {
        "Lazy": {
            "Study": 0.8,
            "Pass Exam": 0.2
        },
        "Productive": {
            "Study": 1.0
        }
    },
    "More Sleep": {
        "Lazy": {
            "More Sleep": 1.0,
        },
        "Productive": {
            "Study": 0.5,
            "More Sleep": 0.5
        }
    },
    "Study": {
        "Lazy": {
            "More Sleep": 1.0
        },
        "Productive": {
            "Pass Exam": 0.9,
            "Study": 0.1
        }
    },
    "Pass Exam": {
        "Lazy": {},
        "Productive": {}
    }
}

T_LAZY = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0.8, 0.2],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

T_PRODUCTIVE = np.array([
    [0.7, 0, 0.3, 0, 0, 0],
    [0, 0, 0.6, 0.4, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0.5, 0.5, 0],
    [0, 0, 0, 0, 0.1, 0.9],
    [0, 0, 0, 0, 0, 0],
])

REWARD = np.array([[reward(s)] for s in TRANSITIONS.keys()])

STATE_VALUE = [{k: 0 for k in TRANSITIONS.keys()} for _ in range(T+1)]

def reward(state, action) -> float:
    return 1.0 if state == "Pass Exam" else -1.0

def back_step(step):
    return ALPHA*(REWARD + np.matmul(T_LAZY, STATE_VALUE[step]))

def state_value():
    step = T-1
    while step >= 0:
        sv = back_step(step)

        # update STATE_VALUE
        for i, state in enumerate(TRANSITIONS.keys()):
            print(state)
            print(i)
            STATE_VALUE[step][state] = sv[i]

        step -= 1


def action_value(state, action):
    pass

if __name__ == "__main__":
    state_value()

    import json
    print(json.dumps(STATE_VALUE, indent=4))