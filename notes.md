### To do

- Test centering images

- resnet50 pretrained with an additional Dense layer

- When your model saturate with Adam, turn to SGD at a smaller learning rate

- Why not just resize the image with transform without cropping?

- Create two branches, main and deploy

### Issues

- When resizing and croping image in order for them to be squared and the same size, sometimes the animal dissapear. It would be good to explore centering the image, admitting different size images or something else.

- CUDA out of memory. The issue was that a certain GPU was busy, we had to select another device.

- RGBA images inside dataset. When processing images make sure to save them in RGB

- How to choose the correct arquitecture?

- What transformations to do to the image?

- It is impractical to do the transformation directly when defining the data because you cannot calculate the mean and std of only the training split, because it does not exist yet, it is a random split.

- When normalize, use mean and std of only training set? What happens to the testing or inference image, do we also normalize it using that mean and std that we trained the model with? 

- How to split training, validation and testing sets. Usage of library to split folders.

- How to save mean and std values? use the same as for training when infering?