# Day 36 - PyTorch with TensorBoard

explored how to integrate **TensorBoard** with **PyTorch** to visualize training metrics, model architecture, and input images. TensorBoard provides a powerful dashboard for monitoring model performance in real-time.

---

##  What is TensorBoard?

TensorBoard is a visualization toolkit originally developed for TensorFlow but is now compatible with PyTorch through `torch.utils.tensorboard`. It provides interactive dashboards to view:

* **Scalars** (loss, accuracy, etc.)
* **Images** (input, output, weights, etc.)
* **Graph** (model architecture)
* **Histograms**, **Text**, **Embeddings**, etc.

---

###  Setting Up TensorBoard in PyTorch

used the `SummaryWriter` from `torch.utils.tensorboard` to log scalar values, images, and the computational graph:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/day36_tensorboard')
```

* This creates a directory `runs/day36_tensorboard` which stores the logs.

---

### Visualizing Scalars (e.g. Training Loss)

logged the training loss using:

```python
writer.add_scalar('training loss', loss.item(), epoch)
```

* Shows up in the **Scalars** tab.
* **Two curves**:

  * Raw values (thin/faint line)
  * Smoothed average (bold line)
* You can control smoothing with the slider.

---

###  Visualizing Input Images

used torchvision’s `make_grid` to log batches of input images:

```python
img_grid = torchvision.utils.make_grid(images)
writer.add_image('input_images', img_grid)
```

* Shows in the **Images** tab of TensorBoard.
* Useful for checking preprocessing and input pipeline.

---

### Visualizing Model Architecture

added the model graph to TensorBoard with:

```python
writer.add_graph(model, images)
```

* Visualizes the neural network computational graph.
* Appears in the **Graphs** tab.
* Helps understand input/output flow and layer connectivity.

---

### Launching TensorBoard

To view the logs:

```bash
tensorboard --logdir=runs
```

Then open browser: [http://localhost:6006](http://localhost:6006)

---


##  Summary

| Component       | Purpose                                   |
| --------------- | ----------------------------------------- |
| `SummaryWriter` | Main interface for logging to TensorBoard |
| `.add_scalar()` | Log scalar values like loss, accuracy     |
| `.add_image()`  | Log input images for visual inspection    |
| `.add_graph()`  | Log model architecture                    |


