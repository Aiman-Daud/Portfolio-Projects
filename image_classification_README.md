# Image Classification: Pretrained ResNet18 vs Baseline

Fine-tunes a pretrained ResNet18 in PyTorch on a CIFAR-10 subset (5,000
train / 2,000 test images) and compares it against a logistic-regression
baseline on raw pixels.

## Results

+------------------------------+----------------------+
| Model                        | Test accuracy        |
+==============================+======================+
| Logistic regression (raw     | 31.8%                |
| pixels, baseline)            |                      |
+------------------------------+----------------------+
| ResNet18 fine-tuned (3       | 80.6%                |
| epochs)                      |                      |
+------------------------------+----------------------+
| Confusion matrix                                    |
+-----------------------------------------------------+

: ![](media/image1.png){width="5.4375in" height="5.4375in"}

The fine-tuned ResNet18 improved on the baseline by about 49 percentage
points.

**Observations:** The most frequent errors were between visually similar
classes: dog and cat (27 and 24 misclassifications), airplane and ship
(19), and truck and automobile (14 and 11). Vehicle classes performed
best (automobile F1 0.89, ship 0.88, truck 0.87), while cat (0.71) and
dog (0.72) were the weakest

## Next steps

Data augmentation, more epochs, the full 50,000-image dataset, and
trying object detection.
