import torch

import sonata
import sonata.data

# organise data needed for sonata
point = torch.load("predictions_sonata.pt")
point["coord"] = point["coord"].numpy()  # Ensure coordinates are float
# Load the pre-trained model from Huggingface
# supported models: "sonata"
# ckpt is cached in ~/.cache/sonata/ckpt, and the path can be customized by setting 'download_root'
model = sonata.model.load("sonata", repo_id="facebook/sonata").cuda()

# Run inference
transform = sonata.transform.default()
point = transform(point)
for key in point.keys():
    if isinstance(point[key], torch.Tensor):
        point[key] = point[key].cuda(non_blocking=True)
pred = model(point)
pred["coord"] = point["coord"]  # Ensure coordinates are float
# Save the predictions to a file
path = "segmentation-sonata.pt"
torch.save(pred, path)
print(f"Predictions saved to {path}")
print(pred.keys())
