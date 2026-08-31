# RPS-King

RPS-King is a desktop application that lets you play Rock, Paper, Scissors against a learning algorithm powered by a custom neural network.

## Overview
Instead of picking random moves, the game follows an algorithm that analyzes your playstyle and tried to predict what you will play next. This prediction is based on your recent match history, move frequencies, and reactions to previous rounds. After feeding this data into the predictor, it passes through layers of "neurons" until giving the final answer. After a round is played, the AI uses a process called backpropagation where it will adjust the connections between all the neurons to see which should be strengthened or weakened based on how they affect the output. Through this process it is able to learn from its mistakes to play better.

To keep you on your toes, the bot employs and epsilon-greedy strategy. This means that 90% of the time, it will counter the move it's thinks you will play, but 10% of the time, it will choose a random option so its behavior isn't completely predictable.

## Features

- Play against an adaptive Ai built entirely from scratch in python
- See how the AI anticipated your next move with the probability bars
- See how it catches onto patterns
- Learns via backpropagation after every round
- View the statistics showing the bot win percentage, draw percentage, and your personal win rate
- Advanced feature extraction that tracks your N-gram move history, outcome transitions, and overall play frequencies
- Presented in an arcade-style pygame interface with a retro look
- Epsilon-greedy strategy (90% prediction usage, 10% random choice) to ensure that the bot remains unpredictable
- Background threading for predictions and learning updates to ensure the visual interface never lags while you play
- Taunts from the AI

<img width="957" height="574" alt="image" src="https://github.com/user-attachments/assets/7d683a3f-00a0-4820-a767-ef3610245dd2" />

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/cream-of-wheat/RPS-King.git
  cd RPS_King
  ```
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Run the application:
  ```bash
  python main.py
  ```

## Usage

* Click on the rock to choose rock
* Click on the paper to choose paper
* Click on the scissors to choose scissors
* Watch the human prediction bars on the right side of the outcome screen to see how the algorithm predicted you would play
* Press any key to return from the outcome screen to the playing screen and play again

## Acknowledgements

* Read the OFL.txt for the license on the retro font used
* The image assets used in this project were sourced a while ago, and I unfortunately no longer have the original credits. If you are the creator of any of these images, please do not hesitate to contact me so I can add proper attribution or remove them upon request.
