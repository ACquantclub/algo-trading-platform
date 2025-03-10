from grader import Grader
import json

if __name__ == "__main__":
    grader = Grader()
    results = grader.grade_submission()

    with open("/autograder/results/results.json", "w") as f:
        json.dump(results, f, indent=2)
