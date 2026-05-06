import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

# source: https://medium.com/data-science-in-your-pocket/getting-started-with-dinov2-installation-setup-and-inference-made-easy-be37b07d32e7
# https://huggingface.co/docs/transformers/en/model_doc/dinov2

MODEL_NAME = "facebook/dinov2-large"
# MODEL_NAME = "facebook/dinov2-base"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu" # use gpu if available

# load pretrained model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME) # image processor for DINOv2 (resizes and normalizes images)
backbone  = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE) # download ViT model weights
backbone.eval()

for p in backbone.parameters(): # no need for gradients, just inference
    p.requires_grad = False

# embed a single image
def embed_image(path):
    img = Image.open(path).convert("RGB") # open image and convert to RGB
    inputs = processor(images=img, return_tensors="pt") # run through processor, returns dict of Pytorch tensors
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()} # {pixels_values: tensor}

    with torch.no_grad():
        # run inputs through model, returns dict with keys: last_hidden_state, pooler_output, hidden_states, attentions
        out = backbone(**inputs) # unpack inputs from dict

    return out.last_hidden_state[:, 0, :].squeeze() # gets CLS token from last hidden state

# embed test images
IMG_DIR = os.path.join("./data")
test_files = [f for f in os.listdir(IMG_DIR)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print("Computing embedding for all test images\n")
test_embeddings = {}
for fname in sorted(test_files):
    path = os.path.join(IMG_DIR, fname)
    emb = embed_image(path)
    test_embeddings[fname] = emb

# cosine similarity between all pairs
names = sorted(test_embeddings.keys())
embs = torch.stack([test_embeddings[n] for n in names])
embs = F.normalize(embs, dim=1) # normalize to unit vectors
sim = (embs @ embs.T).cpu().numpy() # cosine similarity, dot product of normalized vectors

print("\nCosine similarity between all image pairs\n")
for n in names:
    print(f"{n[:10]:>12}", end="") # print all names as column headers
print()
for i, n in enumerate(names):
    print(f"{n:<25}", end="") # print all names as row headers
    for j in range(len(names)):
        print(f"{sim[i,j]:>12.3f}", end="") # get similarity score for images i and j
    print()
