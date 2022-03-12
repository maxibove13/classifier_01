# Animal Classifier

Implementation of a classifier using the [Animals-10](https://www.kaggle.com/alessiocorrado99/animals10) dataset from [kaggle](kaggle.com)

We tested as models simple CNN and a resnet18, best results so far obtained with [resnet18](https://pytorch.org/vision/main/generated/torchvision.models.resnet18.html)

![Loss and accuracy](https://github.com/maxibove13/classifier_01/blob/main/figures/loss_acc_evol.png?raw=true)
## Test deployed API:

Run a POST request to [https://animal-classifier01.herokuapp.com/](https://animal-classifier01.herokuapp.com/) containing an image from the following categories:

categories = ['sheep', 'cat', 'cow', 'butterfly', 'dog', 'squirrel', 'chicken', 'spider', 'elephant', 'horse']

## Training instructions

0. Create virtual environment and install necessary modules with `pip`:

Create venv:

```
python3 -m venv .venv
```

Activate it:

```
. .venv/bin/activate
```

```
pip install -r requirements-dev.txt
```

1. Run `split_data.py` to split the processed dataset in 80% for training, 10% for validation and 10% for final testing. You can change the ratios as you wish.

```
python3 ./app/split_data.py --ratio 0.8 0.1 0.1
```

2. Run `process_images.py` in order to resize and crop raw images (in order for them to be of the same square size) and generate csv file with image filename and associated label (animal type)

```
python3 ./app/process_images.py --size 256 --set train
python3 ./app/process_images.py --size 256 --set val
python3 ./app/process_images.py --size 256 --set test
```

We choose 256x256 px, but that can be changed.

3. Run training script

```
train_model.py --model <model>
````

It is recommended to run the script in the background and throw the prints in a log file, like this:

```
python3 ./app/train_model.py --model <model>  > ./logs/<log_file> &
```

Where <model> could be either a simple `cnn` or `resnet18` loaded from `torchvision`


4. Test model

Run `test_model.py` to test the model in `train`, `val` or `test` sets

```
python3 ./app/test_model.py --model <model> --set <set>
```

5. Inference

Make predictions on random images from `test` set using `infer.py` script.

```
python3 ./app/infer.py --model resnet18 --samples <num_samples>
```

## Instructions to deploy API to Heroku

1. Login with Heroku

```
heroku login -i
```

2. Create heroku app

```
heroku create animal-classifier
```

3. Test heroku locally

```
heroku local
```

4. Associate heroku app with git repository

```
heroku git:remote -a animal_classifier01
```

5. Push to heroku

```
git push heroku main
```
