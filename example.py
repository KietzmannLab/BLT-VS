import argparse
import torch
from blt_vs_model import blt_vs_model, get_blt_vs_transform, load_class_names
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

def main(dataset):
    # Load the class names for the specified dataset
    class_names = load_class_names(dataset=dataset)

    # Load the model with pre-trained weights
    model = blt_vs_model(pretrained=True, training_dataset=dataset)
    model.eval()

    # Get the required transforms
    transform = get_blt_vs_transform()

    # Load a local image
    image_path = 'turtle.jpg'  # Replace with your local image path
    image = Image.open(image_path)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    plt.show()

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Process the output
    final_output = output[-1]  # Get the output from the last timestep
    probabilities = torch.softmax(final_output, dim=1)
    _, predicted_class = torch.max(probabilities, dim=1)
    print(f'Predicted class index: {predicted_class.item()}')

    # Map the predicted class index to the class name
    predicted_class_name = class_names[predicted_class.item()]
    print(f'Predicted class: {predicted_class_name}')

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run image classification with the BLT-VS model.")
    parser.add_argument(
        '--training_dataset',
        type=str,
        default='imagenet',
        help="Model trained on which dataset (e.g., 'imagenet'). Default is 'imagenet'."
    )
    args = parser.parse_args()

    # Call the main function with the provided dataset
    main(dataset=args.training_dataset)