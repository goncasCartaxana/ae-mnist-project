"""
The Trainer class encapsulates the training loop for an AutoEncoder model.
It handles the training and testing processes, and computes the VAE loss.

Saves the results:
- model.pth              # Model of last epoch
- best_model.pth         # Model at best epoch (lowest validation loss)
- metrics.json           # Summary: config, train_losses, test_losses, final/best
- training_history.csv   # Per-epoch losses (ready for Excel/pandas/matplotlib)
"""

from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, config: dict):
        print("Starting trainer...")
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # Either: cuda(nvidia gpu), mps(apple), cpu
        print(f"Device used: {self.device}")
        self.model.to(self.device)
        self.best_test_loss = float('inf') 

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 1e-3) # default = 0.001
        )

        self.train_losses = []
        self.test_losses = []

        self.loss_type = config.get("loss_type", "reconstruction")  # e.g. "reconstruction" ou "vae" (reconstruction + KL)

    # ===== LOSSES =====

    def reconstruction_loss(self, x, x_hat):
        # MSE ou BCE
        return F.mse_loss(x_hat, x, reduction="sum")

    def vae_loss(self, x, x_hat, mean, log_var):
        # VAE loss = reconstruction (BCE) + KL divergence
        bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return bce + kld
    
    def compute_loss(self, x, model_output):
        """
        Accepts:
        - input: "AE" -> output = x_hat
        - input "VAE" -> model_output = (x_hat, mean, log_var)
        """

        if self.loss_type == "reconstruction":
            x_hat = model_output
            return self.reconstruction_loss(x, x_hat)

        elif self.loss_type == "vae":
            x_hat, mean, log_var = model_output
            return self.vae_loss(x, x_hat, mean, log_var)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    # ===== TRAIN / TEST =====

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for x, _ in tqdm(dataloader, desc="Train", leave=False):
            x = x.to(self.device).view(x.size(0), -1)
            self.optimizer.zero_grad()
            model_output = self.model(x)
            loss = self.compute_loss(x, model_output)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader.dataset)

    def test_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, _ in tqdm(dataloader, desc="Test", leave=False):
                x = x.to(self.device).view(x.size(0), -1)
                model_output = self.model(x)
                loss = self.compute_loss(x, model_output)
                total_loss += loss.item()

        return total_loss / len(dataloader.dataset)

    def train(self, train_loader, test_loader):
        """Train model and save best_model.pth when validation improves."""
        epochs = self.config.get("epochs", 10)

        self.best_test_loss = float('inf')

        for epoch in range(epochs):
            # Train + test
            train_loss = self.train_epoch(train_loader)
            test_loss = self.test_epoch(test_loader)

            # Saves losses
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            # Saves new model if best
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                exp_dir = Path(self.config.get('exp_dir', 'results'))

                # Create Directory if needed
                results_dir = exp_dir / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'best_loss': test_loss,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                }, results_dir / "best_model.pth")
                print(f"Best improved: {test_loss:.4f} (epoch {epoch+1})")

            # Print progress
            print(
                f"[{epoch+1:2d}/{epochs}] "
                f"train={train_loss:.4f} "
                f"test={test_loss:.4f} "
                f"best={self.best_test_loss:.4f}"
            )
         
        # Final print
        print(f"Training concluded! Best loss: {self.best_test_loss:.4f}")


    def save_results(self, exp_dir: Path):
        """Save all results (models + metrics) with clear console output."""
        results_dir = exp_dir / "results"
        results_dir.mkdir(exist_ok=True)

        print("\n" + "="*80)
        print("SALVING FINAL RESULTS!")
        print("="*80)

        # 1. Best model (saved during training)
        best_path = results_dir / "best_model.pth"
        if best_path.exists():
            print(f"BEST MODEL:  {best_path}")
        else:
            print("Best_model.pth not found")
        
        # 2. Save final model (last epoch)
        torch.save(self.model.state_dict(), results_dir / "model.pth")
        print(f"FINAL MODEL: {results_dir / 'model.pth'}")

        # 3. Full metrics (config + losses + final/best)
        metrics = {
            "config": self.config,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_test_loss": self.test_losses[-1] if self.test_losses else None,
            "best_test_loss": getattr(self, "best_test_loss", None),
            "num_epochs": len(self.train_losses),
        }
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"METRICS:     {results_dir / 'metrics.json'}")

         # 4. Per-epoch CSV
        df = pd.DataFrame({
            "epoch": list(range(1, len(self.train_losses) + 1)),
            "train_loss": self.train_losses,
            "test_loss": self.test_losses,
        })
        df.to_csv(results_dir / "training_history.csv", index=False)
        print(f"CSV:         {results_dir / 'training_history.csv'}")


        # Summary
        print("\nFINAL RESULTS:")
        print(f"    Best test loss: {getattr(self, 'best_test_loss', 'N/A'):.4f}")
        print(f"    Train epochs:   {len(self.train_losses)}")
        print(f"    Test epochs:    {len(self.test_losses)}")

        if self.train_losses:
            print(f"    Final train:   {self.train_losses[-1]:.4f}")
        if self.test_losses:
            print(f"    Final test:    {self.test_losses[-1]:.4f}")
        print("=" * 80 + "\n")

