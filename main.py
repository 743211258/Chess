import chess
import torch
import torch.optim as optim
import chess.engine
from torch import nn
from evaluation import evaluation
from cnn import CNN
from input import Train, board_to_tensor

# parameters and constants
batch_size = 100
batch_inputs = []
batch_labels = []
model = CNN(kernel_size=3, num_kernel_in_first_layer=13, num_kernel_in_second_layer=64, padding=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.MSELoss()

# extract the data from pgn file
data = Train(r"C:\Chess\lichess_db_standard_rated_2013-01.pgn", max_games=10000)
boards = data.extract_data()

engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\ericz\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")

for epoch in range(1):
    total_loss = 0
    for i, board in enumerate(boards):
        if i % 1000 == 0:
            print(f"已处理第 {i} 个局面")
        board_tensor = board_to_tensor(board)
        label = evaluation(board, engine)

        batch_inputs.append(board_tensor)
        batch_labels.append(label)

        if len(batch_inputs) == batch_size:
            board_tensor_group = torch.stack(batch_inputs)
            label_tensor = torch.tensor(batch_labels, dtype=torch.float32)

            output = model(board_tensor_group).squeeze(1)
            loss = loss_function(output, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_inputs = []
            batch_labels = []
    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(boards)}")
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': total_loss / len(boards),
}, 'model_checkpoint.pth')
print("Shutting down the program.")
engine.quit()
print("Program shuts down successfully.")