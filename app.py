from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms
from model import Generator
import os
import time
import random

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load both generators
G_A2B = Generator().to(device)
G_A2B.load_state_dict(torch.load("checkpoints/netG_A2B_epoch9.pth", map_location=device))
G_A2B.eval()

G_B2A = Generator().to(device)
G_B2A.load_state_dict(torch.load("checkpoints/netG_B2A_epoch9.pth", map_location=device))
G_B2A.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        direction = request.form.get('direction', 'A2B')

        if img_file:
            img = Image.open(img_file).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                if direction == 'A2B':
                    output_tensor = G_A2B(input_tensor)
                else:
                    output_tensor = G_B2A(input_tensor)

            output_img = transforms.ToPILImage()(output_tensor.squeeze(0) * 0.5 + 0.5)

            os.makedirs("static", exist_ok=True)
            timestamp = int(time.time())
            output_path = f'static/output_{timestamp}.jpg'
            output_img.save(output_path)

            return render_template(
                'index.html',
                result=output_path,
                direction=direction,
                timestamp=timestamp
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
