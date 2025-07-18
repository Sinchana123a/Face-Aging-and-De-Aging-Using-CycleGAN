Face Aging and De-Aging using CycleGAN:
This project uses a CycleGAN model trained on the UTKFace dataset to transform human face images by simulating aging and de-aging effects. Users can upload a face image through a Flask web interface and get a transformed image as output.

Features:
CycleGAN-based image-to-image translation
Face aging (young → old) and de-aging (old → young)
Trained from scratch on UTKFace dataset (no pretrained models used)
Simple web interface using Flask

How to Run:
1.Clone the Repository
2.git clone https://github.com/SahanaReddy06/faceAgingandDeAgingUsingCycleGAN.git
3.cd "FaceAging and DeAging Using CycleGAN"
4 Install Dependencies
  pip install -r requirements.txt
5.Download UTFFace Dataset and place in the floder of UTFFace
  Now Run This Command python split_utkface.py that splits the dataset into a trainA and trainB
6.Add Model Checkpoint
Place your trained model file (e.g., faceAging_epoch10.pth) into the checkpoints/ folder, for this run python train.py
6.Run the App
python app.py
Open your browser at http://127.0.0.1:5000

Notes:
The model handles both aging and de-aging using separate generators.
Make sure uploaded images are clear and frontal for best results.
Outputs are saved in the results/ directory.

#Note
I deleted dataset beacuse of large size, so please download UTKFace Dataset and follow steps 