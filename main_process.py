from dino_agent import DinoAgent, train_network, init_cache, build_model
from game import Game, Game_sate


# main function
def playGame(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_sate(dino, game)
    model = build_model()
    try:
        train_network(model, game_state, observe=observe)
    except StopIteration:
        game.end()


if __name__ == "__main__":
    init_cache()
    # playGame(observe=False)
