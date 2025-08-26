import torch
from tqdm import tqdm


from components.generator import Generator
from components.discriminator import Discriminator
from components.trainer import CycleGANTrainer
from components.dataPrepare import train_loader, val_loader, test_loader

# === Config ===
BATCH_SIZE = 1
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Initialize Models ===
G = Generator().to(DEVICE)   #G: A → B
F = Generator().to(DEVICE)   #F: B → A
D_A = Discriminator().to(DEVICE)  #Disc for domain A
D_B = Discriminator().to(DEVICE)  #Disc for domain B

# === Trainer ===
trainer = CycleGANTrainer(G, F, D_A, D_B, device=DEVICE)

# === Training Loop ===
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for i, batch in enumerate(loop):
        real_A, real_B = batch["A"].to(DEVICE), batch["B"].to(DEVICE)
        trainer.set_input(real_A, real_B)
        g_loss, dA_loss, dB_loss = trainer.train_step()

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(G=g_loss, D_A=dA_loss, D_B=dB_loss)

    # === validation after each epoch ===
    with torch.no_grad():
        val_g_losses, val_dA_losses, val_dB_losses = [], [], []
        for batch in val_loader:
            real_A, real_B = batch["A"].to(DEVICE), batch["B"].to(DEVICE)
            g_loss, dA_loss, dB_loss = trainer.evaluate_step(real_A, real_B)
            val_g_losses.append(g_loss)
            val_dA_losses.append(dA_loss)
            val_dB_losses.append(dB_loss)
        print(f"Validation -> G: {sum(val_g_losses)/len(val_g_losses):.4f}, "
              f"D_A: {sum(val_dA_losses)/len(val_dA_losses):.4f}, "
              f"D_B: {sum(val_dB_losses)/len(val_dB_losses):.4f}")

    #save models every 10 epochs
    if (epoch+1) % 10 == 0:
        torch.save(G.state_dict(), f"G_epoch{epoch+1}.pth")
        torch.save(F.state_dict(), f"F_epoch{epoch+1}.pth")
        torch.save(D_A.state_dict(), f"D_A_epoch{epoch+1}.pth")
        torch.save(D_B.state_dict(), f"D_B_epoch{epoch+1}.pth")
        print(f"Saved models at epoch {epoch+1}")

# === Final Testing after training ===
with torch.no_grad():
    test_g_losses, test_dA_losses, test_dB_losses = [], [], []
    for batch in test_loader:
        real_A, real_B = batch["A"].to(DEVICE), batch["B"].to(DEVICE)
        g_loss, dA_loss, dB_loss = trainer.evaluate_step(real_A, real_B)
        test_g_losses.append(g_loss)
        test_dA_losses.append(dA_loss)
        test_dB_losses.append(dB_loss)

    print(f"Test -> G: {sum(test_g_losses)/len(test_g_losses):.4f}, "
          f"D_A: {sum(test_dA_losses)/len(test_dA_losses):.4f}, "
          f"D_B: {sum(test_dB_losses)/len(test_dB_losses):.4f}")

# === Plot training & validation losses ===
trainer.plot_losses()