# MAGIC Telescope Gamma Ray Classification

This project utilizes machine learning techniques to classify high-energy particles—specifically gamma rays and hadrons—detected by the MAGIC (Major Atmospheric Gamma-ray Imaging Cherenkov) telescope. The goal is to develop a predictive model that can accurately distinguish between these two types of particles based on their observed properties.

## Dataset

The dataset employed is the MAGIC Gamma Telescope dataset, which simulates the detection of high-energy gamma particles using a ground-based atmospheric Cherenkov gamma telescope. It comprises 19,020 events, each characterized by 10 continuous features and a binary class label:

- **fLength**: Major axis of the ellipse [mm]
- **fWidth**: Minor axis of the ellipse [mm]
- **fSize**: 10-log of the sum of content of all pixels [in #photons]
- **fConc**: Ratio of the sum of two highest pixels over fSize [ratio]
- **fConc1**: Ratio of the highest pixel over fSize [ratio]
- **fAsym**: Distance from the highest pixel to the center, projected onto the major axis [mm]
- **fM3Long**: Third root of the third moment along the major axis [mm]
- **fM3Trans**: Third root of the third moment along the minor axis [mm]
- **fAlpha**: Angle of the major axis with the vector to the origin [deg]
- **fDist**: Distance from the origin to the center of the ellipse [mm]
- **class**: Target variable with two categories: 'g' for gamma rays and 'h' for hadron rays

The dataset file is named `magic04.data` and is included in this repository.

## Project Structure

- `main.py`: Contains the implementation of the machine learning model(s) used for classification.
- `magic04.data`: The dataset file containing the features and labels.
- `MAGIC_TELESCOPE_PREDICTION.pptx`: A presentation detailing the project's methodology and results.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running the Project

1. Clone the repository:

```bash
git clone https://github.com/Sree2k3/Magic-Telescope.git
cd Magic-Telescope
```

2. Run the main script:

```bash
python main.py
```

This will execute the classification model and output the results, including performance metrics and visualizations.

## Methodology

The project explores various machine learning algorithms to classify the particles, such as:

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines
- K-Nearest Neighbors

Model performance is evaluated using metrics like accuracy, precision, recall, and F1-score. Cross-validation techniques are employed to ensure the robustness of the models.

## Results

The presentation file `MAGIC_TELESCOPE_PREDICTION.pptx` provides a comprehensive overview of the project's findings, including:

- Data preprocessing steps
- Exploratory data analysis
- Model selection and tuning
- Performance comparison of different algorithms
- Conclusions and future work

## Citation

If you use this dataset or project in your research, please cite:

Bock, R. et al. (2007). MAGIC Gamma Telescope. UCI Machine Learning Repository. [https://doi.org/10.24432/C52C8B](https://doi.org/10.24432/C52C8B)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
