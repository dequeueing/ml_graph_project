"""Compatibility entry point for the current audit, including exclusions.
The earlier first-pass scorer is archived under before-false-negative-review.
"""
from review_scores import main
if __name__ == "__main__":
    main()
