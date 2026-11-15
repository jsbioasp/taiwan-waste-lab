# Course notebooks

Copies of the notebooks from the AI & Python course that the [prep weeks](../docs/PREREQS.md) point back to. Each one opens straight in [Google Colab](https://colab.research.google.com/) from the links below. Use **File → Save a copy in Drive** to keep your own editable version.

**Most notebooks run as they are.** The M01–M04 notebooks download what they need: public datasets like MNIST and CIFAR-10, or pretrained models.

**The M05 and M06 notebooks need the course data bundles** (`petfinder-mini.zip`, `aac-mini.zip`, `m06-photos.zip`). The bundles also contain the helper code some of these notebooks import. They aren't in this repo, because the PetFinder data comes from a Kaggle competition with its own terms. Ask your instructor for the Google Drive links, and paste them where the notebook says `CONFIGURE_ME`. Without the bundles you can still read these notebooks as lessons.

These are teaching copies. The live course versions may get updates, and fixes made here don't flow back.

| Notebook | What it covers | Prep week | Needs a data bundle? |
|---|---|---|---|
| [M01-6_numpy](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M01-6_numpy.ipynb) | NumPy arrays | D | No |
| [M01-lab5_numpy_matplotlib](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M01-lab5_numpy_matplotlib.ipynb) | Arrays and plotting with matplotlib | D | No |
| [M02-3_linear_regression](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M02-3_linear_regression.ipynb) | Linear regression and MSE | D | No |
| [M02-lab2_linear_regression](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M02-lab2_linear_regression.ipynb) | Training and evaluating a linear regression | D | No |
| [M02-6_evaluation](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M02-6_evaluation.ipynb) | Precision, recall, F1, confusion matrices | A, B | No |
| [M03-lab3_mnist_evaluation](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M03-lab3_mnist_evaluation.ipynb) | Digging into a model's mistakes | A | No |
| [M04-2_cnn_building_blocks](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-2_cnn_building_blocks.ipynb) | Convolution, pooling and other CNN building blocks | B | No |
| [M04-3_cnn_architectures](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-3_cnn_architectures.ipynb) | From LeNet to ResNet, including skip connections | B | No |
| [M04-4_pretrained_models](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-4_pretrained_models.ipynb) | Using pretrained models | A, C | No |
| [M04-lab2_build_a_cnn](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-lab2_build_a_cnn.ipynb) | Building a CNN for CIFAR-10 | C, refresher | No |
| [M04-lab3_cnn_experiments](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-lab3_cnn_experiments.ipynb) | Experimenting with a CNN | C, refresher | No |
| [M04-lab4_pretrained_features](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-lab4_pretrained_features.ipynb) | Pretrained models and feature maps | C | No |
| [M04-deep_transfer_learning](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-deep_transfer_learning.ipynb) | Fine-tuning a pretrained model: frozen vs fully trained | A, C | No |
| [M04-deep_data_augmentation](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-deep_data_augmentation.ipynb) | Data augmentation | A, C | No |
| [M05-4_meet_aac](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M05-4_meet_aac.ipynb) | A second dataset, and why models fail to transfer | B | `aac-mini` |
| [M05-lab4_cross_dataset](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M05-lab4_cross_dataset.ipynb) | Cross-dataset experiments, hands-on | B | `petfinder-mini`, `aac-mini` |
| [M05-lab5_audit_your_model](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M05-lab5_audit_your_model.ipynb) | Group-aware cross-validation and leakage | A, D | `petfinder-mini`, `aac-mini` |
| [M06-1_photos_and_adoption](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-1_photos_and_adoption.ipynb) | Pretrained CNN features as model inputs | C | `petfinder-mini`, `m06-photos` |
| [M06-lab1_photo_features](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-lab1_photo_features.ipynb) | Extracting photo features, hands-on | C | `petfinder-mini`, `m06-photos` |
| [M06-3_multitask_with_photos](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-3_multitask_with_photos.ipynb) | Shared trunk, separate heads, auxiliary target | B | `petfinder-mini`, `m06-photos` |
| [M06-lab2_multitask_handson](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-lab2_multitask_handson.ipynb) | Multi-task training, hands-on | B | `petfinder-mini`, `m06-photos` |
| [M06-5_taiwan_deployment](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-5_taiwan_deployment.ipynb) | Checking a model against official numbers (Spearman) | D | `petfinder-mini`, `aac-mini` |
| [M06-lab3_taiwan_handson](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-lab3_taiwan_handson.ipynb) | Taiwan deployment, hands-on | D | `petfinder-mini`, `aac-mini` |
