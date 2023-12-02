import random
import sys
from dvclive import Live


# we are tracking the accuracy and loss of both train and val
# we are also tracking the number of epochs
# we are also saving the dvclive experiments
with Live(save_dvc_exp=True) as live:
    epochs = 10
    live.log_param("epochs", epochs)
    for epoch in range(epochs):
        live.log_metric("train/accuracy", epoch + random.random())
        live.log_metric("train/loss", epochs - epoch - random.random())
        live.log_metric("val/accuracy",epoch + random.random() )
        live.log_metric("val/loss", epochs - epoch - random.random())
        live.next_step()