import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor_to_pil(tensor):
    image = tensor.cpu().clone().squeeze(0)
    return transforms.ToPILImage()(image)

# ── Loss modules ──────────────────────────────────────────────────────────────

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# ── Normalization (lives inside the model so images stay in [0,1]) ────────────

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std  = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std  = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# ── Model builder ─────────────────────────────────────────────────────────────

content_layers_default = ['conv_4']
style_layers_default   = ['conv_1', 'conv_2', 'conv_3']   # tuned version

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                style_img, content_img,
                                content_layers=content_layers_default,
                                style_layers=style_layers_default):
    normalization  = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses   = []
    model          = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name  = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'batchNorm_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target       = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss     = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    # Trim layers after the last loss module
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

# ── Main transfer function ────────────────────────────────────────────────────

def run_style_transfer(content_path, style_path,
                       num_steps=300,
                       style_weight=30_000_000,
                       content_weight=3,
                       progress_callback=None):
    """
    Run neural style transfer and return the output PIL image.

    progress_callback(step, total, style_loss, content_loss) called every 50 steps.
    """
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
    for p in cnn.parameters():
        p.requires_grad_(False)

    content_img = image_loader(content_path)
    style_img   = image_loader(style_path)
    input_img   = content_img.clone()

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        style_img, content_img
    )
    model.eval()
    model.requires_grad_(False)

    input_img.requires_grad_(True)
    optimizer = optim.LBFGS([input_img])

    run = [0]

    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)

            style_score   = sum(sl.loss for sl in style_losses)   * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0 and progress_callback:
                progress_callback(run[0], num_steps,
                                  style_score.item(), content_score.item())
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return tensor_to_pil(input_img)
