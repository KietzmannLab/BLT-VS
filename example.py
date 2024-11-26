import torch
from blt_vs_model import blt_vs_model, get_blt_vs_transform, load_class_names
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Load the model with pre-trained weights
dataset = 'imagenet'
# Load the class names for the specified dataset
class_names = load_class_names(dataset=dataset)

model = blt_vs_model(pretrained=True, training_dataset=dataset)
model.eval()

# Get the required transforms
transform = get_blt_vs_transform()

# Download an example image
image_url = 'https://upload.wikimedia.org/wikipedia/commons/e/ea/Baby_turtle.jpg'  # Replace with a valid image URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

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