# Face Mask Type Detection using Multiclass Classification

This repository contains Python code for a face mask type detection project using multiclass classification. The project aims to classify different types of face masks, including cloth face masks, N95 face masks, N95 face masks with valves, surgical face masks, and images with no face masks. The dataset used for training and evaluation consists of over 2000 clean images with a resolution of 300x300 pixels.

## Dataset

The dataset used for training and evaluation can be found on Kaggle: [Face Mask Types Dataset](https://www.kaggle.com/datasets/bahadoreizadkhah/face-mask-types-dataset). It contains images categorized into the following classes:

- `cloth`: Cloth face mask images
- `n95`: N95 face mask images
- `n95v`: N95 with Valve face mask images
- `nfm`: No Face Mask images
- `srg`: Surgical face mask images

The train and test sets are already separated in the dataset.

## Dependencies

To run the code in this repository, you'll need the following dependencies:

- Python 3.x
- OpenCV
- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the required packages using `pip`:

```
pip install opencv-python tensorflow keras numpy matplotlib
```

## Usage

1. Clone this repository to your local machine:

```
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Download the Face Mask Types dataset from the provided link and place it in the appropriate directory.

3. Run the script to predict the mask type from an uploaded image:

```
python predict_image.py --image path/to/your/image.jpg
```

4. Run the script to predict the mask type from a live video stream:

```
python predict_video.py
```

Make sure to replace `path/to/your/image.jpg` with the actual path to your desired image file.

## Results

The trained model can accurately classify the type of face mask with the provided dataset. You can modify the code and experiment with different architectures or hyperparameters to potentially improve the performance.

## Acknowledgments

- The Face Mask Types dataset used in this project was sourced from Kaggle: [Face Mask Types Dataset](https://www.kaggle.com/datasets/bahadoreizadkhah/face-mask-types-dataset).

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute to the project by submitting pull requests or suggesting improvements. If you encounter any issues or have questions, please open an issue in the repository.
