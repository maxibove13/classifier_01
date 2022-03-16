# Animal Classifier

Implementation of a multi-class classifier using the [Animals-10](https://www.kaggle.com/alessiocorrado99/animals10) dataset from [kaggle](kaggle.com)

This repository contains the offline training of the model using PyTorch, an API implementing the inference of the trained model and a basic front-end interface that the client can interact with.

Go to [https://whichisit.netlify.app/](https://whichisit.netlify.app/) to make predictions on new animals

categories = ['sheep', 'cat', 'cow', 'butterfly', 'dog', 'squirrel', 'chicken', 'spider', 'elephant', 'horse']
# training

We tested as models simple CNN and a resnet18, best results so far obtained with [resnet18](https://pytorch.org/vision/main/generated/torchvision.models.resnet18.html)

![Loss and accuracy](https://github.com/maxibove13/classifier_01/blob/main/api/figures/loss_acc_evol.png?raw=true)

## Training instructions

1. Create virtual environment and install necessary modules with `pip`:

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

2. Run `split_data.py` to split the processed dataset in 80% for training, 10% for validation and 10% for final testing. You can change the ratios as you wish.

    ```
    python3 ./app/split_data.py --ratio 0.8 0.1 0.1
    ```

3. Run `process_images.py` in order to resize and crop raw images (in order for them to be of the same square size) and generate csv file with image filename and associated label (animal type)

    ```
    python3 ./app/process_images.py --size 256 --set train
    python3 ./app/process_images.py --size 256 --set val
    python3 ./app/process_images.py --size 256 --set test
    ```

    We choose 256x256 px, but that can be changed.

4. Run training script

    ```
    train_model.py --model <model>
    ```

    It is recommended to run the script in the background and throw the prints in a log file, like this:

    ```
    python3 ./app/train_model.py --model <model>  > ./logs/<log_file> &
    ```

    Where <model> could be either a simple `cnn` or `resnet18` loaded from `torchvision`


5. Test model

    Run `test_model.py` to test the model in `train`, `val` or `test` sets

    ```
    python3 ./app/test_model.py --model <model> --set <set>
    ```
6. Inference

    Make predictions on random images from `test` set using `infer.py` script.

    ```
    python3 ./app/infer.py --model resnet18 --samples <num_samples>
    ```
# api

Based on the trained model we implemented a Rest API using Flask to create and endpoint that makes predictions on a new image of an animal. 
The API is deployed in heroku and this [endpoint](https://animal-classifier01.herokuapp.com/) only accepts POST requests with an image file in the body: 

## Instructions to deploy API to heroku

You can either deploy the app with Heroku CLI or in Heroku [dashboard](https://dashboard.heroku.com/)

### Heroku CLI:

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
# client

In order for a user to make predictions on new images we implemented a basic ReactJS [app](https://whichisit.netlify.app/) as our project frontend.

## Create React app

1. Create react app

    With `node` and `npm` installed:

    ```
    npx create-react-app whichisit
    ```

2. Start a development server:

    ```
    npm start
    ```

2. Set an environment variable with the API endpoint

    i. Create a `.env` file with:

    ```
    REACT_APP_API = "https://animal-classifier01.herokuapp.com/infer"
    ```

3. Implement a fetch `POST` request to the endpoint:

    ```js
    fetch(`${process.env.REACT_APP_API}`, {
                method: "POST",
                body: image_data
            })
    ```

4. When you are ready to deploy to production create `build` with `npm`:

    ```
    npm run build
    ```

4. Deploy using [Netlify](https://www.netlify.com/)

    We found easier to deploy a React app in Netlify than in Heroku.

    i. In the Netlify dashboard change the following Build settings (note that the React root directory is a subfolder of our repository)

    ```
    Base directory: client
    Build command: npm run build
    Publish directory: client/build
    ```