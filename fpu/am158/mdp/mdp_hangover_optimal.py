import math
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

def policy_eval(step, T, alpha, STATE_VALUES, ACTION_VALUES):
    # set up state/action value results
    for s in STATES:
        STATE_VALUES[s] = [0]*(T+1)
        for a in ACTIONS:
            ACTION_VALUES[f"{s}-{a}"] = [0]*(T+1)
    
    for i in range(T-1, step, -1):
        for s in STATES:
            for a in ACTIONS: 
                transitions = TRANSITIONS[s][a]

                # calculate Q for each s-a pair
                reward = get_reward(s, a)
                q_ev = sum([
                    p*(
                        # hardcoded for two actions + alpha probability
                        alpha*ACTION_VALUES[f"{state}-Lazy"][i+1]+
                        (1-alpha)*ACTION_VALUES[f"{state}-Productive"][i+1]
                    ) for state, p in transitions.items()
                ])

                ACTION_VALUES[f"{s}-{a}"][i] = reward + q_ev

            STATE_VALUES[s][i] = alpha*ACTION_VALUES[f"{s}-Lazy"][i] + (1-alpha)*ACTION_VALUES[f"{s}-Productive"][i]

    return STATE_VALUES, ACTION_VALUES


def find_optimal_policy(T, alpha, OPT_SV, OPT_AV, OPT_POL):
    for s in STATES:
        OPT_SV[s] = [0]*(T+1)
        OPT_POL[s] = ["None"]*(T+1)
        for a in ACTIONS:
            OPT_AV[f"{s}-{a}"] = [0]*(T+1)

    for i in range(T-1, -1, -1):
        for s in STATES:
            # calc max av first -- technically less space efficient (O(|S||A|)), but we need these values anyway + more time efficient
            curr_max = -(math.inf)
            max_action = "None"
            for a in ACTIONS:
                OPT_AV[f"{s}-{a}"][i] = get_reward(s, a) + sum(
                    [p*(OPT_SV[state][i+1]) for state, p in TRANSITIONS[s][a].items()]
                )
                if OPT_AV[f"{s}-{a}"][i] > curr_max:
                    curr_max = OPT_AV[f"{s}-{a}"][i]
                    max_action = a

            # calc max sv reusing the max av calculations
            OPT_SV[s][i] = curr_max
            OPT_POL[s][i] = max_action

    return OPT_SV, OPT_AV, OPT_POL

    

if __name__ == "__main__":
    sv, av = policy_eval(-1, 10, 0.4, {}, {})
    opt_sv, opt_av, opt_pol = find_optimal_policy(10, 0.4, {}, {}, {})

    import json
    # print(json.dumps(sv, indent=2))
    # print(json.dumps(av, indent=2))
    print("--- Optimal state values (t = 0) ---")
    print([
        opt_sv[s][0] for s in STATES
    ])
    print("--- Optimal policy (t = 0) ---\n")
    print([
        opt_pol[s][0] for s in STATES
    ])
