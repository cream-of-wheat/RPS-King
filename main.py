from enum import Enum
import random
import math
import pygame
import threading


class Predictor:
    def __init__(self, num_games, learning_rate = 0.1, num_hidden_layers=1, momentum=0.85):
        self.learning_rate = learning_rate
        self.num_games = num_games
        self.momentum = momentum
        self.layers = []

        for i in range(num_hidden_layers):
            if not i:
                self.layers.append(Layer(24, num_games * 6))
            else:
                self.layers.append(Layer(24, 24))

        self.layers.append(Layer(3, 24, last_layer=True))

        self.prediction = None

    def forward(self, inputs):
        self.prediction = None

        new_inputs = []

        for game_input in inputs:
            for i in range(6):
                new_inputs.append(game_input[i])

        for layer in self.layers:
            new_inputs = layer.forward(new_inputs)

        self.prediction = Predictor.softmax(new_inputs)

        return self.prediction

    def backward(self, target):
        delta = [self.prediction[i] - target[i] for i in range(3)]

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        for layer in self.layers:
            layer.update(self.learning_rate, self.momentum)

    @staticmethod
    def softmax(vector):
        max_vector = max(vector)
        stab_vector = [vector_comp - max_vector for vector_comp in vector]
        exp_vector = [math.exp(stab_comp) for stab_comp in stab_vector]
        exp_sum = sum(exp_vector)

        return [exp_comp/exp_sum for exp_comp in exp_vector]


class Layer:
    def __init__(self, num_neurons, num_inputs, last_layer=False): # last layer is different from the rest. All others need the ReLu
        self.last_layer = last_layer
        self.neurons = [Neuron(num_inputs, self.last_layer) for _ in range(num_neurons)]
        self.inputs = []

    def forward(self, inputs):
        self.inputs = inputs
        outputs = []

        for neuron in self.neurons:
            output = neuron.forward(inputs)

            if not self.last_layer:
                if output < 0:
                    output *= 0.01

            neuron.output = output

            outputs.append(output)

        return outputs

    def backward(self, delta):
        for i in range(len(self.neurons)):
            if not self.last_layer and self.neurons[i].output <=0:
                self.neurons[i].delta = delta[i] * 0.01
            else:
                self.neurons[i].delta = delta[i]

        next_layer_deltas = [0.0 for _ in range(len(self.inputs))]

        for neuron in self.neurons:
            for j in range(len(neuron.weights)):
                next_layer_deltas[j] += neuron.weights[j] * neuron.delta

        return next_layer_deltas

    def update(self, learning_rate, momentum=0.85):
        for neuron in self.neurons:
            neuron.update(self.inputs, learning_rate, momentum)

class Neuron:
    def __init__(self, num_inputs, last_layer=False):
        if last_layer:
            std_dev = math.sqrt(1.0 / num_inputs)
        else:
            std_dev = math.sqrt(2.0 / num_inputs)

        self.weights = [random.gauss(0.0, std_dev) for _ in range(num_inputs)]
        self.bias = 0.01

        self.velocity = [0.0 for _ in range(num_inputs)]
        self.velocity_bias = 0.0

        self.raw_output = 0.0
        self.output = 0.0
        self.delta = 0.0

    def forward(self, inputs):
        self.raw_output = self.bias

        for i in range(len(inputs)):
            self.raw_output += self.weights[i] * inputs[i]

        return self.raw_output

    def update(self, inputs, learning_rate, momentum = 0.85):
        for i in range(len(self.weights)):
            gradient = inputs[i] * self.delta
            # self.weights[i] -= learning_rate * gradient
            self.velocity[i] = (momentum * self.velocity[i]) - (learning_rate * gradient)
            self.weights[i] += self.velocity[i]

        self.velocity_bias = (momentum * self.velocity_bias) - (learning_rate * self.delta)
        self.bias += self.velocity_bias
        # self.bias -= self.delta * learning_rate


class Manager:
    def __init__(self, num_games = 6, learning_rate_bot = 0.06, momentum=0.40):
        self.num_games = num_games
        self.learning_rate_bot = learning_rate_bot
        self.momentum = momentum
        # self.games = [random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) + random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) for _ in range(self.num_games)]
        self.games = [[0, 0, 0, 0, 0, 0] for _ in range(self.num_games)]
        self.predictor = Predictor(self.num_games, learning_rate=self.learning_rate_bot, momentum=self.momentum)
        self.predictions = []

    def add_new_round(self, choice):
        del self.games[0]
        self.games.append(choice)

    def predict(self):
        self.predictions = self.predictor.forward(self.games)


        # Greedy threshold strategy:
        if max(self.predictions) > 0.45 or random.random() < 0.80:
            return self.predictions.index(max(self.predictions))
        else:
            return random.choices([0, 1, 2])[0]

        # return random.choices([0, 1, 2], weights=self.predictions, k=1)[0]

    def learn(self, target_choice):
        self.predictor.backward(target_choice)


class Outcome(Enum):
    PLAYER_WIN = 0
    DRAW = 1
    BOT_WIN = 2


class Button:
    def __init__(self, x, y, width, height, normal_color=(64, 64, 64), hover_color = (84, 84, 84)):
        self.rect = pygame.Rect(x, y, width, height)
        self.normal_color = normal_color
        self.hover_color = hover_color

        self.hover = False

    def draw(self, surface):
        if self.hover:
            color = self.hover_color
        else:
            color = self.normal_color

        pygame.draw.rect(surface, color, self.rect)

    def check_hover(self, mouse_pos):
        self.hover = self.rect.collidepoint(mouse_pos)
        return self.hover

    def clicked(self, mouse_pos, event):
        if self.rect.collidepoint(mouse_pos):
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                return True

        return False


class GameState(Enum):
    CHOICE_SCREEN = 1
    OUTCOME_SCREEN = 2


class Game:
    def __init__(self):
        self.manager = Manager()

        self.bot_wins = 0
        self.draws = 0
        self.human_wins = 0

        self.convert_answer = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.convert_to_array = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}

        self.game_state = GameState.CHOICE_SCREEN
        self.player_choice = None
        self.prediction = None
        self.next_prediction = self.manager.predict()
        self.is_thinking = False
        self.predictions = self.manager.predictions
        self.result = None

        pygame.init()
        self.screen = pygame.display.set_mode((960, 540))
        self.font = pygame.font.Font("assets/PressStart2P.ttf", 24)
        pygame.display.set_caption("Rock, Paper, Scissors")
        self.clock = pygame.time.Clock()

        self.animation_phase = 0
        self.len_animation = 2
        self.animation_time = 0.3
        self.animation_time_left = self.animation_time

        self.buttons = {
            "rock": Button(40, 340, 160, 160),
            "paper": Button(240, 340, 160, 160),
            "scissors": Button(440, 340, 160, 160)
        }

        self.images = {
            "rock": [pygame.image.load("assets/rock.png").convert_alpha(), pygame.image.load("assets/rock-attack.png").convert_alpha()],
            "paper": [pygame.image.load("assets/paper.png").convert_alpha(), pygame.image.load("assets/paper-attack.png").convert_alpha()],
            "scissors": [pygame.image.load("assets/scissors.png").convert_alpha(), pygame.image.load("assets/scissors-attack.png").convert_alpha()],
            "computer": pygame.transform.scale_by(pygame.image.load("assets/computer.png").convert_alpha(), 2)
        }

        self.scanlines = pygame.Surface((960, 540), pygame.SRCALPHA)
        for y in range(0, 540, 3):
            pygame.draw.line(self.scanlines, (0, 0, 0,  60), (0, y), (960, y), 1)

        self.taunts = {
            Outcome.BOT_WIN: ["PREDICTABLE", "TOO EASY", "IS THAT ALL YA GOT?"],
            Outcome.PLAYER_WIN: ["LUCKY GUESS", "IMPOSSIBLE", "BEGINNER'S LUCK"],
            Outcome.DRAW: ["STALEMATE", "I'M GOING EASY", "LOGIC TIE"]
        }
        self.current_taunt = ""

        self.running = True

    def run(self):
        while self.running:
            time_delta = self.clock.tick(60) / 1000.0
            self.update_animation(time_delta)

            self.handle_events()

            if self.game_state == GameState.OUTCOME_SCREEN:
                self.draw_outcome()
            else:
                self.draw_choice()

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons: self.buttons.get(button).check_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.MOUSEBUTTONUP and self.game_state == GameState.CHOICE_SCREEN:
                if self.buttons.get("rock").clicked(mouse_pos, event):
                    self.player_choice = "rock"
                if self.buttons.get("paper").clicked(mouse_pos, event):
                    self.player_choice = "paper"
                if self.buttons.get("scissors").clicked(mouse_pos, event):
                    self.player_choice = "scissors"

                if self.player_choice:
                    self.game_state = GameState.OUTCOME_SCREEN
                    self.prediction = self.next_prediction
                    self.predictions = self.manager.predictions
                    player_choice = self.convert_answer.get(self.player_choice)
                    self.result = Game.get_result(player_choice, self.prediction)
                    self.update_stats(self.result)

                    self.is_thinking = True
                    threading.Thread(
                        target=self.update_and_predict,
                        args=(player_choice, self.prediction),
                        daemon=True
                    ).start()

            if event.type == pygame.KEYDOWN and self.game_state == GameState.OUTCOME_SCREEN:
                self.player_choice = None
                self.prediction = None
                self.result = None
                self.current_taunt = ""
                self.game_state = GameState.CHOICE_SCREEN

    def update_and_predict(self, plyr_choice, bot_choice):
        self.update_manager(plyr_choice, bot_choice)
        self.next_prediction = self.manager.predict()
        self.is_thinking = False

    def draw_choice(self):
        self.screen.fill((25, 25, 85))

        for button in self.buttons: self.buttons.get(button).draw(self.screen)

        self.screen.blit(self.images.get("rock")[self.animation_phase], (43, 343))
        self.screen.blit(self.images.get("paper")[self.animation_phase], (243, 343))
        self.screen.blit(self.images.get("scissors")[self.animation_phase], (443, 343))

        self.screen.blit(self.images.get("computer"), (720, 180))
        self.draw_stats()

        self.screen.blit(self.scanlines, (0, 0))
        pygame.display.flip()

    def draw_outcome(self):
        self.screen.fill((25, 25, 85))

        pygame.draw.rect(self.screen, (64, 64, 64), (40 + 200 * self.convert_answer.get(self.player_choice), 340, 160, 160))
        self.screen.blit(self.images.get(self.player_choice)[0], (43 + 200 * self.convert_answer.get(self.player_choice), 343))

        pygame.draw.rect(self.screen, (64, 64, 64), (720, 340, 160, 160))
        self.screen.blit(self.images.get(list(self.images)[self.prediction])[0], (723, 343))
        self.screen.blit(self.images.get("computer"), (720, 180))
        self.draw_prob_bars(self.predictions)
        self.draw_stats()

        if pygame.time.get_ticks() % 1000 < 500:
            text = self.font.render("PRESS ANY KEY TO PLAY AGAIN", True, (255, 255, 0))
            self.screen.blit(text, (40, 280))

        if self.current_taunt:
            taunt_text = self.font.render(self.current_taunt, True, (0, 0, 0))
            text_rect = taunt_text.get_rect(midright=(640, 200))

            pygame.draw.rect(self.screen, (255, 255, 255), text_rect.inflate(40, 20))
            pygame.draw.rect(self.screen, (0, 255, 255), text_rect.inflate(40, 20), 4)

            pygame.draw.polygon(self.screen, (255, 255, 255), [(700, 200), (680, 190), (680, 210)])
            pygame.draw.lines(self.screen, (0, 255, 255), False, [(680, 190), (700, 200), (680, 210)], 4)

            self.screen.blit(taunt_text, text_rect)

        self.screen.blit(self.scanlines, (0, 0))
        pygame.display.flip()

    def draw_prob_bars(self, predictions):
        labels = ["Rock", "Paper", "Scissors"]

        x = 520
        y = 55
        bar_width = 150
        bar_height = 18
        spacing = 28

        for i, probability in enumerate(predictions):
            label = labels[i]
            percent = probability * 100

            text = self.font.render(f"{label}", True, (255, 255, 255))
            text_rect = text.get_rect(topright=(x+130, y + i * spacing))
            self.screen.blit(text, text_rect)

            bar_x = x + 150
            bar_y = y + i *spacing

            pygame.draw.rect(self.screen, (220, 220, 220), (bar_x, bar_y, int(bar_width * probability), bar_height))

            percent_text = self.font.render(f"{percent:.1f}%", True, (0, 255, 255))
            self.screen.blit(percent_text, (bar_x + bar_width + 15, bar_y - 2))

    def draw_stats(self):
        total_games = self.bot_wins + self.draws + self.human_wins

        if total_games == 0:
            bot_percent = 0
            draw_percent = 0
            human_percent = 0
        else:
            bot_percent = self.bot_wins / total_games * 100
            draw_percent = self.draws / total_games * 100
            human_percent = self.human_wins / total_games * 100

        stats = [
            f"Bot win%:{bot_percent:.1f}%",
            f"Draw%:{draw_percent:.1f}%",
            f"Human win%:{human_percent:.1f}%"
        ]

        x = 40
        y = 40
        spacing = 30

        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (255, 255, 255))
            self.screen.blit(text, (x, y + i * spacing))

        text = self.font.render(f"Total games: {total_games}", True, (255, 255, 255))
        self.screen.blit(text, (40, 130))

    @staticmethod
    def get_result(plyr_choice, bot_choice):
        if plyr_choice == bot_choice:
            return Outcome.DRAW
        elif (plyr_choice + 1) % 3 == bot_choice:
            return Outcome.BOT_WIN
        else:
            return Outcome.PLAYER_WIN

    def update_stats(self, result):
        if result == Outcome.PLAYER_WIN:
            self.human_wins += 1
            print("you won")
        elif result == Outcome.BOT_WIN:
            self.bot_wins += 1
            print("computer won")
        else:
            self.draws += 1
            print("tie")

        self.current_taunt = random.choice(self.taunts[result])

        # total_games = self.bot_wins + self.draws + self.human_wins
        # print(f"Bot win%: {self.bot_wins / total_games * 100}%; draw%: {self.draws / total_games * 100}%; Human win%: {self.human_wins / total_games * 100}%; Total games: {total_games}")

    def update_manager(self, plyr_choice, bot_choice):
        plyr_choice_arr = self.convert_to_array.get(plyr_choice)
        target_choice = self.convert_to_array.get((plyr_choice + 1) % 3)
        bot_choice_arr = self.convert_to_array.get(bot_choice)

        self.manager.add_new_round(plyr_choice_arr + bot_choice_arr)
        self.manager.learn(target_choice)

    def update_animation(self, time):
        self.animation_time_left -= time

        if self.animation_time_left <= 0:
            self.animation_time_left += self.animation_time
            self.animation_phase = (self.animation_phase + 1) % self.len_animation


if __name__ == "__main__":
    game = Game()
    game.run()
