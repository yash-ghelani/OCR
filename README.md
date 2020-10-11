# Developing an Optical Character Recognition system

## Objective

- To build and evaluate an optical character recognition system that can process scanned book pages and turn them into text.

### To test the code:

Ensure you are in the correct directory then run:

pip install -r requirements.txt

Followed by:

run_evaluate.sh

The code should print out the percentage of correctly classified characters for each page.

### Overall performance

The figures below indicate accuracy/performance using nearest neighbour and PCA based approach - Each page decreases in quality with added noise.

Noise | Score
--- | ---
1 | 98%
2 | 98%
3 | 92%
4 | 78%
5 | 63%
6 | 51%
