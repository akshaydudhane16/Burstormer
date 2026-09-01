import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("version_0/metrics.csv")

# PL's CSVLogger writes one row per logged step, with many columns NaN
# depending on what was logged that step (train vs val). Split them out:
train = df[["step", "train_loss_step"]].dropna()
val = df[["step", "val_psnr"]].dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train["step"], train["train_loss_step"])
axes[0].set_title("Train loss")
axes[0].set_xlabel("step")

axes[1].plot(val["step"], val["val_psnr"])
axes[1].set_title("Val PSNR")
axes[1].set_xlabel("step")

plt.tight_layout()
plt.savefig("progress.png", dpi=150)
print("saved progress.png")