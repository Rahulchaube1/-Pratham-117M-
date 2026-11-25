"""Reproducible short-answer regression checks for the Pratham chat release.

Pass a callable that accepts a list of role/content messages and returns only
the new assistant reply. Keep these prompts out of instruction-tuning data.
"""
import re
import unicodedata

CASES = [
    ("extract-blue", "Mira put the key in a blue box. What color is the box? Answer in one word.", {"blue"}),
    ("extract-red", "Mira put the key in a red box. What color is the box? Answer in one word.", {"red"}),
    ("extract-owner", "Isha gave her umbrella to Dev. Who received the umbrella? Answer with a name.", {"dev"}),
    ("addition", "What is two plus two? Answer with just the result.", {"4", "four"}),
    ("subtraction", "I had seven pears and gave two away. How many remain? Answer with a number.", {"5", "five"}),
    ("negation", "The vase fell but did not break. Did it break? Answer yes or no.", {"no"}),
    ("missing", "The parcel arrived yesterday. What color is it? If unspecified, answer unknown.", {"unknown"}),
]

def normalize(text):
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()

def evaluate(reply):
    results = []
    for name, prompt, expected in CASES:
        answer = reply([{"role": "user", "content": prompt}])
        results.append(dict(case=name, prompt=prompt, answer=answer,
                            passed=normalize(answer) in expected))
    history = [{"role": "user", "content": "My bag is green."},
               {"role": "assistant", "content": "Understood."},
               {"role": "user", "content": "Correction: it is purple now."},
               {"role": "assistant", "content": "Understood."},
               {"role": "user", "content": "What color is my bag now? Answer in one word."}]
    answer = reply(history)
    results.append(dict(case="multi-turn-correction", prompt=history, answer=answer,
                        passed=normalize(answer) == "purple"))
    return results
