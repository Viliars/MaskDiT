import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
from movqgan import get_movqgan_model


def prepare_image(pil_image):
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))


def main(args):
    assert torch.cuda.is_available()

    model = get_movqgan_model('270M', pretrained=True, device='cuda')

    root = ""

    images = sorted(os.listdir(f"{root}/ffhq256_fid"))

    batch_size = 50

    for i in tqdm(range(0, 70000, batch_size)):
        bufer = []
        names = []
        for j in range(i, i+batch_size):
            image_path = f"{root}/ffhq256_fid/{images[j]}"

            bufer.append(prepare_image(Image.open(image_path)).unsqueeze(0))

        batch = torch.cat(bufer, dim=0)
        
        with torch.no_grad():
            latents = model.encode(batch.to('cuda'))[0]

        latents = latents.to('cpu').numpy()
        for z, j in enumerate(range(i, i+batch_size)):
            np.save(f"{root}/latents_mo/{j:05d}.npy", latents[z])
