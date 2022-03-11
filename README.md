# Animal Classifier

Implementation of a classifier using the [Animals-10](https://www.kaggle.com/alessiocorrado99/animals10) dataset from [kaggle](kaggle.com)

Best results so far using `resnet18`:

![Loss and accuracy](https://github.com/maxibove13/classifier_01/blob/main/figures/loss_acc_evol.png?raw=true)

## Instructions

1. Run `split_data.py` to split the processed dataset in 80% for training, 10% for validation and 10% for final testing. You can change the ratios as you wish.
```
python3 ./src/split_data.py --ratio 0.8 0.1 0.1
```

2. Run `process_images.py` in order to resize and crop raw images and generate csv file with image filename and associated label (animal type)

```
python3 ./src/process_images.py --size 256 --set train
python3 ./src/process_images.py --size 256 --set val
```

We will process the `test` set later on, as this set will not be used until the model is trained and validated.


2. Run 

```
train_model.py --model <model>
````

It is recommended to run the script in the background and throw the prints in a log file, like this:

```
python3 ./src/train_model.py --model <model>  > ./logs/<log_file> &
```

Where <model> could be either a simple `cnn` or `resnet18` loaded from `torchvision`