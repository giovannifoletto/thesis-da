import os
import torch
from torch import optim, nn, utils, Tensor
import lightning as L

# define any number of nn.Modules (or use your current ones)
# define the LightningModule
class LitDrainLSTM(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()

        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
                nn.LSTM(
                    input_size=self.input_size, 
                    hidden_size=self.hidden_size, 
                    batch_first=True
                ), 
                #nn.ReLU(), 
                nn.Linear(64, 1)
            )
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.model(x, (h0, c0))
        out = self.fc(out[:, 1, :])
        return out


    def training_step(self, batch, batch_idx):
        model.train(True)
        print(f"Epoch: {epoch+1}")
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
          x_batch, y_batch = batch[0].to(device), batch[1].to(device)

          output = model(x_batch)
          loss = loss_function(output, y_batch)
          running_loss += loss.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss/100
            print("Batch {0}, Loss {1:.5f}".format(
                batch_index+1,
                avg_loss_across_batches
            ))
            running_loss = 0.0


        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)