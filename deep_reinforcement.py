import torch
import torch.optim as optim
import torch.nn as nn
import chess
from input import board_to_tensor
from cnn import CNN
from mcts import MCTS

def self_play_game(model, simulations_per_move=50, max_moves=80):
    board = chess.Board()
    states = []

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        mcts = MCTS(model, simulations=simulations_per_move)
        move = mcts.search(board)
        if move is None:
            break
        states.append((board.copy(), None))  # Label to be filled after game
        board.push(move)

    result = board.result()
    if result == '1-0':
        value = 1.0
    elif result == '0-1':
        value = -1.0
    else:
        value = 0.0

    data = []
    for board_pos, _ in states:
        data.append((board_to_tensor(board_pos), torch.tensor(value, dtype=torch.float32)))
        value *= -1  # Viewpoint flip

    return data

def simulate_games(model, num_games=10, simulations_per_move=30):
    training_data = []
    for game_index in range(num_games):
        game_data = self_play_game(model, simulations_per_move=simulations_per_move)
        training_data.extend(game_data)
        print(f"  Game {game_index + 1}: collected {len(game_data)} states")
    return training_data

def train_model(model, optimizer, loss_function, num_epochs=10, num_games_per_epoch=10, simulations_per_move=30):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        training_data = simulate_games(model, num_games=num_games_per_epoch, simulations_per_move=simulations_per_move)

        total_loss = 0
        model.train()
        for tensor, label in training_data:
            input_tensor = tensor.unsqueeze(0)  # Shape: [1, 13, 8, 8]
            output = model(input_tensor).squeeze()
            loss = loss_function(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save the model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, 'model_checkpoint.pth')

    print("Training complete and model saved.")

def main():
    # Initialize the model and optimizer
    model = CNN(kernel_size=3, num_kernel_in_first_layer=13, num_kernel_in_second_layer=32, padding=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.MSELoss()

    # Start training
    try:
        train_model(model, optimizer, loss_function, num_epochs=10, num_games_per_epoch=10, simulations_per_move=30)
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
