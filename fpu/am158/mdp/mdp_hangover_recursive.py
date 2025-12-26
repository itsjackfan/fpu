import numpy as np

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
    "More Sleep": {
        "Lazy": {
            "More Sleep": 1.0,
        },
        "Productive": {
            "Study": 0.5,
            "More Sleep": 0.5
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
        "Lazy": {
            "Pass Exam": 1.0
        },
        "Productive": {
            "Pass Exam": 1.0
        }
    }
}

def get_reward(state, action) -> float:
    return 1.0 if state == "Pass Exam" else -1.0

# recursive state value
def state_value(T, ALPHA, step):
    STATE_VALUE = [{k: 0 for k in TRANSITIONS.keys()} for _ in range(T+1)]

    for i in range(T - 1, step, -1):
        # print(f"Currently on step {i}")
        # print(STATE_VALUE[i])
        for k, v in TRANSITIONS.items():
            # print(f"Now evaluating {k}")
            # lazy calculation
            lazy_state_ev = sum([
                p*(STATE_VALUE[i+1][state]) for state, p in v["Lazy"].items()
            ])
            # print("Lazy EV: ", lazy_state_ev)
            lazy_value = ALPHA*(
                get_reward(k, "Lazy") + lazy_state_ev
            )
            # print("Lazy value: ", lazy_value)

            # productive calculation
            productive_state_ev = sum([
                p*(STATE_VALUE[i+1][state]) for state, p in v["Productive"].items()
            ])
            # print("Productive EV: ", productive_state_ev)
            productive_value = (1-ALPHA)*(
                get_reward(k, "Lazy") + productive_state_ev
            )
            # print("Productive value: ", productive_value)

            STATE_VALUE[i][k] = lazy_value + productive_value

    return STATE_VALUE


def action_value(T, ALPHA, step, sv):
    state_action_pairs = []
    for s in STATES:
        for a in ACTIONS:
            state_action_pairs.append(f"{s}-{a}")

    ACTION_VALUE = [{k: 0 for k in state_action_pairs} for _ in range(T+1)]

    for i in range(T - 1, step, -1):
        # print(f"Currently on step {i}")
        # print(STATE_VALUE[i])
        current_values = ACTION_VALUE[i]

        for pair in current_values.keys():
            # print(f"Now evaluating {k}")
            state = pair.split("-")[0]
            action = pair.split("-")[1]

            # reward
            reward = get_reward(state, action)

            # ev of state value
            curr_space = TRANSITIONS[state][action]
            sv_ev = sum(
                [v*sv[i+1][state] for v in curr_space.values()]
            )
            
            current_values[pair] = reward + sv_ev

    return ACTION_VALUE

if __name__ == "__main__":
    T = 10
    ALPHA = 0.4

    sv = state_value(T, ALPHA, -1)
    av = action_value(T, ALPHA, -1, sv)

    import json
    print(json.dumps(sv, indent=4))
    print(json.dumps(av, indent=4))