# Developing an Optical Character Recognition system

## Objective

- To build and evaluate an optical character recognition system that can process scanned book pages and turn them into text.

### Overall performance

The figures below indicate accuracy/performance using nearest neighbour and PCA based approach - Each page decreases in quality with added noise.

| Noise value 0 - 98% accuracy         | Noise value 0.1 - 98% accuracy       | Noise value 0.2 - 92% accuracy       |
| ------------------------------------ | ------------------------------------ | ------------------------------------ |
| ![](https://i.imgur.com/bMLCvJg.png) | ![](https://i.imgur.com/3jUF4qz.png) | ![](https://i.imgur.com/em1rniO.png) |

| Noise value 0.3 - 78% accuracy       | Noise value 0.4 - 63% accuracy       | Noise value 0.5 - 51% accuracy       |
| ------------------------------------ | ------------------------------------ | ------------------------------------ |
| ![](https://i.imgur.com/eOGaAhh.png) | ![](https://i.imgur.com/yvnjrVm.png) | ![](https://i.imgur.com/9bAKi5O.png) |

### To test the code:

Ensure you are in the correct directory then run:

pip install -r requirements.txt

Followed by:

run_evaluate.sh

The code should print out the percentage of correctly classified characters for each page.
