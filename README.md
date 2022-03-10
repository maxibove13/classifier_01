# Animal Classifier

Implementation of a classifier using the [Animals-10](https://www.kaggle.com/alessiocorrado99/animals10) dataset from [kaggle](kaggle.com)

Best results so far using `resnet18`:

![Loss and accuracy](https://github.com/maxibove13/classifier_01/blob/main/figures/loss_acc_evol.png?raw=true)

## Instructions

1. Run `process_images.py` in order to resize and crop raw images and generate csv file with image filename and associated label (animal type)

2. Run 

```
train_model.py --model <model>
````

Where <model> could be either a simple `cnn` or `resnet18` loaded from `torchvision`