"""
Fast Neural Style Transfer — Maximum Quality Training

Accuracy improvements:
  1. Five VGG style layers       relu1_2/2_2/3_3/4_3/5_3
  2. Feature statistics loss     per-channel mean + std matching
  3. Identity loss               net(style) ≈ style every N steps
  4. Multi-scale style grams     averaged over 3 scales
  5. Per-layer style weights     tunable per layer
  6. LR warm-up + cosine decay
  7. AMP, grad-clip, augmentation, 9 res-blocks @ 288px

  8. ★ Patch adversarial loss    70×70 PatchGAN discriminator
     The discriminator learns to tell real style patches from generated ones.
     The generator is forced to produce micro-textures the discriminator cannot
     distinguish from the real style — far beyond what gram matrices enforce.
     Spectral norm on D + one-sided label smoothing for training stability.

Pause:   touch training.pause  /  rm training.pause
Stop:    Ctrl+C  →  checkpoint auto-saved
Resume:  python train.py ... --resume models/<name>_resume.pth
"""

import os, sys, signal, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
from pathlib import Path

from transformer_net import TransformerNet, PatchDiscriminator

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP    = device.type == 'cuda'
PAUSE_FILE = 'training.pause'
EXTS       = {'.jpg', '.jpeg', '.png', '.webp'}
os.makedirs('models', exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, root, size, manifest=None):
        if manifest and os.path.isfile(manifest):
            with open(manifest) as f:
                self.paths = [Path(l.strip()) for l in f if l.strip()]
            print(f'Loaded {len(self.paths):,} paths from manifest: {manifest}')
        else:
            self.paths = sorted([p for p in Path(root).rglob('*') if p.suffix.lower() in EXTS])
            if manifest:
                print(f'Manifest not found at {manifest}, scanning directory...')

        if not self.paths:
            raise ValueError(f'No images found in {root}')

        self.transform = transforms.Compose([
            transforms.Resize(size + 64),           # wider crop margin for more variety
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15,
                saturation=0.15, hue=0.03),          # stronger jitter → better generalisation
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            return self.transform(Image.open(self.paths[idx]).convert('RGB'))
        except Exception:
            return self[0]


class SeededSampler(Sampler):
    def __init__(self, n, epoch, offset=0):
        g = torch.Generator()
        g.manual_seed(epoch)
        self.indices = torch.randperm(n, generator=g).tolist()[offset:]

    def __iter__(self): return iter(self.indices)
    def __len__(self):  return len(self.indices)


def make_loader(dataset, batch_size, epoch, batch_offset=0):
    sampler = SeededSampler(len(dataset), epoch, batch_offset * batch_size)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=4, pin_memory=True, drop_last=True)


# ── VGG feature extractor (5 layers) ─────────────────────────────────────────
#
#  Layer       Index range   Channels  What it captures
#  relu1_2     [:4]          64        fine edges, grain, noise
#  relu2_2     [4:9]         128       strokes, local texture   ← content loss
#  relu3_3     [9:16]        256       mid-level patterns
#  relu4_3     [18:25]       512       shapes, coarse structure
#  relu5_3     [27:34]       512       global composition

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        ch = list(vgg19(weights=VGG19_Weights.DEFAULT).features.eval().children())
        self.s1 = nn.Sequential(*ch[:4])     # relu1_2
        self.s2 = nn.Sequential(*ch[4:9])    # relu2_2
        self.s3 = nn.Sequential(*ch[9:16])   # relu3_3
        self.s4 = nn.Sequential(*ch[18:25])  # relu4_3
        self.s5 = nn.Sequential(*ch[27:34])  # relu5_3
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x  = (x - self.mean) / self.std
        h1 = self.s1(x)
        h2 = self.s2(h1)
        h3 = self.s3(h2)
        # s4/s5 start from max-pool outputs, so chain from h3
        h4 = self.s4(nn.functional.max_pool2d(h3, 2))
        h5 = self.s5(nn.functional.max_pool2d(h4, 2))
        return h1, h2, h3, h4, h5   # indices 0–4


# ── Loss functions ────────────────────────────────────────────────────────────

def gram(x):
    """[B,C,H,W] → [B,C,C] normalised Gram matrix (computed in float32 to avoid AMP overflow)."""
    B, C, H, W = x.shape
    f = x.float().view(B, C, H * W)   # float16 bmm overflows; float32 is safe
    return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)


def gram_style_loss(feat_out, style_grams, batch_size, layer_weights):
    """Weighted gram-matrix style loss across all layers."""
    loss = 0.0
    for fo, sg, w in zip(feat_out, style_grams, layer_weights):
        loss += w * F.mse_loss(gram(fo), sg.expand(batch_size, -1, -1))
    return loss


def feature_stats_loss(feat_out, feat_style, layer_weights):
    """
    Per-channel mean + std matching (complements gram matrices).
    Forces the full feature distribution to match, not just covariances.
    """
    loss = 0.0
    for fo, fs, w in zip(feat_out, feat_style, layer_weights):
        B, C = fo.shape[:2]
        fo_f = fo.float().view(B, C, -1)   # cast for numerical stability under AMP
        fs_f = fs.float().view(1, C, -1)
        loss += w * (F.mse_loss(fo_f.mean(-1), fs_f.mean(-1).expand(B, -1)) +
                     F.mse_loss(fo_f.std(-1),  fs_f.std(-1).expand(B, -1)))
    return loss


def tv_loss(x):
    return (torch.mean(torch.abs(x[:,:,:,:-1] - x[:,:,:,1:])) +
            torch.mean(torch.abs(x[:,:,:-1,:] - x[:,:,1:,:])))


# ── Adversarial losses ────────────────────────────────────────────────────────

def d_loss(disc, real, fake):
    """
    Discriminator loss with one-sided label smoothing.
    Real labels = 0.9 (not 1.0) — prevents D from becoming overconfident,
    which destabilises G gradients early in training.
    """
    d_real = disc(real.detach())
    d_fake = disc(fake.detach())
    loss_real = F.binary_cross_entropy_with_logits(
        d_real, torch.full_like(d_real, 0.9))   # label smoothing
    loss_fake = F.binary_cross_entropy_with_logits(
        d_fake, torch.zeros_like(d_fake))
    return (loss_real + loss_fake) * 0.5


def g_adv_loss(disc, fake):
    """Generator adversarial loss: fool D into classifying fake as real."""
    d_fake = disc(fake)
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def style_real_batch(style_t, batch_size):
    """
    Build a batch of 'real' style samples for D by applying random augmentations
    to the single style image. Variety prevents D from simply memorising one image.
    """
    batch = style_t.expand(batch_size, -1, -1, -1).clone()
    if torch.rand(1).item() > 0.5:
        batch = torch.flip(batch, dims=[3])                     # H-flip
    brightness = 0.85 + torch.rand(1).item() * 0.3             # ×[0.85, 1.15]
    batch = (batch * brightness).clamp(0, 1)
    noise = torch.randn_like(batch) * 0.01                      # tiny noise
    return (batch + noise).clamp(0, 1)


# ── Multi-scale style grams ───────────────────────────────────────────────────

def compute_multiscale_grams(vgg, style_t, scales=(1.0, 0.75, 0.5)):
    """
    Average gram matrices from multiple scales of the style image.
    Makes style transfer robust across different content image sizes.
    """
    all_grams = []
    with torch.no_grad():
        for s in scales:
            img = F.interpolate(style_t, scale_factor=s,
                                mode='bilinear', align_corners=False) if s != 1.0 else style_t
            all_grams.append([gram(f) for f in vgg(img)])

    # Average across scales
    n = len(scales)
    return [sum(ag[i] for ag in all_grams) / n for i in range(len(all_grams[0]))]


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(path, net, disc, optim_g, optim_d, scheduler, scaler, epoch, step, args):
    torch.save({
        'epoch':            epoch,
        'global_step':      step,
        'model_state':      net.state_dict(),
        'disc_state':       disc.state_dict(),
        'optimizer_state':  optim_g.state_dict(),
        'optim_d_state':    optim_d.state_dict(),
        'scheduler_state':  scheduler.state_dict(),
        'scaler_state':     scaler.state_dict(),
        'n_residual':       args.n_residual,
        'args':             vars(args),
    }, path)


def load_checkpoint(path, net, disc, optim_g, optim_d, scheduler, scaler):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt['model_state'])
    if 'disc_state' in ckpt:
        disc.load_state_dict(ckpt['disc_state'])
    optim_g.load_state_dict(ckpt['optimizer_state'])
    if 'optim_d_state' in ckpt:
        optim_d.load_state_dict(ckpt['optim_d_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    if USE_AMP and 'scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state'])
    return ckpt['epoch'], ckpt['global_step']


# ── Pause ─────────────────────────────────────────────────────────────────────

def check_pause():
    if os.path.exists(PAUSE_FILE):
        print(f'\n⏸  Paused — delete "{PAUSE_FILE}" to resume...')
        while os.path.exists(PAUSE_FILE):
            time.sleep(1)
        print('▶  Resuming...\n')


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    dataset = ImageDataset(args.dataset, args.image_size, manifest=args.manifest)

    net    = TransformerNet(n_residual=args.n_residual).to(device)
    disc   = PatchDiscriminator().to(device)
    vgg    = VGGFeatures().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # Separate optimisers — D needs a lower lr to stay behind G
    optim_g = torch.optim.Adam(net.parameters(),  lr=args.lr,      betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(disc.parameters(), lr=args.lr * 0.1, betas=(0.5, 0.999))

    batches_per_epoch = len(dataset) // args.batch_size
    total_steps       = args.epochs * batches_per_epoch

    warmup = min(1000, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim_g,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optim_g, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_g, T_max=max(1, total_steps - warmup), eta_min=args.lr * 0.01),
        ],
        milestones=[warmup],
    )

    # ── Resume ────────────────────────────────────────────────
    start_epoch, global_step = 1, 0
    resume_path = f'models/{args.name}_resume.pth'

    if args.resume:
        if not os.path.isfile(args.resume):
            print(f'ERROR: checkpoint not found: {args.resume}'); sys.exit(1)
        start_epoch, global_step = load_checkpoint(
            args.resume, net, disc, optim_g, optim_d, scheduler, scaler)
        print(f'Resumed from {args.resume}  (ep {start_epoch}, step {global_step})\n')

    # ── Style targets ─────────────────────────────────────────
    to_tensor = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    style_t = to_tensor(Image.open(args.style).convert('RGB')).unsqueeze(0).to(device)

    # Multi-scale gram matrices (averaged over 3 scales)
    style_grams = compute_multiscale_grams(vgg, style_t, scales=(1.0, 0.75, 0.5))

    # Style features at native scale for statistics loss
    with torch.no_grad():
        style_feats = vgg(style_t)

    # Per-layer style weights: emphasise early (texture) layers, soften the deepest one
    layer_weights = [args.sw1, args.sw2, args.sw3, args.sw4, args.sw5]

    # ── TensorBoard + sample dirs ──────────────────────────────
    writer = None
    if _HAS_TB and not args.no_tensorboard:
        os.makedirs('runs', exist_ok=True)
        writer = _TBWriter(log_dir=f'runs/{args.name}', flush_secs=30)

    os.makedirs(f'samples/{args.name}', exist_ok=True)

    if args.sample_image and os.path.isfile(args.sample_image):
        probe_t = to_tensor(Image.open(args.sample_image).convert('RGB')).unsqueeze(0).to(device)
    else:
        probe_t = style_t   # default: style image itself shows identity-loss progress

    # ── Info ──────────────────────────────────────────────────
    n_g = sum(p.numel() for p in net.parameters()) / 1e6
    n_d = sum(p.numel() for p in disc.parameters()) / 1e6
    print(f'Device       : {device}  (AMP={USE_AMP})')
    print(f'Generator    : TransformerNet ({args.n_residual} res-blocks, {n_g:.1f}M params)')
    print(f'Discriminator: PatchGAN ({n_d:.1f}M params, adv-weight={args.adv_weight:.0e})')
    print(f'Dataset      : {len(dataset):,} images → {batches_per_epoch:,} batches/epoch')
    print(f'Total steps  : {total_steps:,}  ({args.epochs} epochs)')
    print(f'Style layers : relu1_2({args.sw1}) relu2_2({args.sw2}) relu3_3({args.sw3}) '
          f'relu4_3({args.sw4}) relu5_3({args.sw5})')
    print(f'Extra losses : feature-stats (w={args.stats_weight})  '
          f'identity every {args.identity_every} steps (w={args.identity_weight})')
    print(f'LR (G)       : {args.lr} →warmup({warmup})→ cosine → {args.lr*0.01}')
    print(f'LR (D)       : {args.lr * 0.1}  (fixed)')
    if writer:
        print(f'TensorBoard  : runs/{args.name}  →  tensorboard --logdir runs')
    print(f'Samples      : samples/{args.name}/  every {args.sample_every} steps')
    print(f'\nPause : touch {PAUSE_FILE}  /  rm {PAUSE_FILE}')
    print(f'Stop  : Ctrl+C  →  {resume_path}\n')

    interrupted = [False]
    def _sigint(sig, frame): interrupted[0] = True
    signal.signal(signal.SIGINT, _sigint)

    # ── Loop ──────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        done_this_epoch = max(0, global_step - (epoch - 1) * batches_per_epoch)
        loader = make_loader(dataset, args.batch_size, epoch, done_this_epoch)
        if done_this_epoch:
            print(f'[ep {epoch}] skipping {done_this_epoch} already-done batches...')

        for batch in loader:
            check_pause()

            if interrupted[0]:
                save_checkpoint(resume_path, net, disc, optim_g, optim_d,
                                scheduler, scaler, epoch, global_step, args)
                print(f'\nSaved → {resume_path}')
                print(f'Resume: python train.py --style {args.style} '
                      f'--dataset {args.dataset} --name {args.name} '
                      f'--resume {resume_path}')
                sys.exit(0)

            batch = batch.to(device)

            # ── Generator forward (AMP for speed) ────────────
            with torch.amp.autocast(device.type, enabled=USE_AMP):
                stylized = net(batch)
                adv_loss = g_adv_loss(disc, stylized)

            # VGG MUST run in float32 — its activations regularly exceed
            # float16 max (~65504), producing inf before gram() is ever called.
            feat_out = vgg(stylized.float())
            with torch.no_grad():
                feat_in = vgg(batch.float())

            c_loss  = F.mse_loss(feat_out[1], feat_in[1])
            s_loss  = gram_style_loss(feat_out, style_grams, args.batch_size, layer_weights)
            fs_loss = feature_stats_loss(feat_out, style_feats, layer_weights)
            t_loss  = tv_loss(stylized.float())

            loss = (args.content_weight * c_loss  +
                    args.style_weight   * s_loss  +
                    args.stats_weight   * fs_loss +
                    args.tv_weight      * t_loss  +
                    args.adv_weight     * adv_loss)

            if global_step % args.identity_every == 0:
                with torch.amp.autocast(device.type, enabled=USE_AMP):
                    id_out = net(style_t)
                id_loss = F.mse_loss(id_out.float(), style_t.float()) * args.identity_weight
                loss    = loss + id_loss

            # ── Generator step ────────────────────────────────
            scaler.scale(loss).backward()
            scaler.unscale_(optim_g)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(optim_g)
            optim_g.zero_grad(set_to_none=True)

            # ── Discriminator step ────────────────────────────
            real_batch = style_real_batch(style_t, batch.size(0))
            with torch.amp.autocast(device.type, enabled=USE_AMP):
                disc_loss = d_loss(disc, real_batch, stylized)

            scaler.scale(disc_loss).backward()
            scaler.unscale_(optim_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
            scaler.step(optim_d)
            scaler.update()
            optim_d.zero_grad(set_to_none=True)

            scheduler.step()
            global_step += 1

            # ── TensorBoard scalars ───────────────────────────
            if writer and global_step % 50 == 0:
                writer.add_scalar('loss/content', c_loss.item(),    global_step)
                writer.add_scalar('loss/style',   s_loss.item(),    global_step)
                writer.add_scalar('loss/stats',   fs_loss.item(),   global_step)
                writer.add_scalar('loss/tv',      t_loss.item(),    global_step)
                writer.add_scalar('loss/adv_G',   adv_loss.item(),  global_step)
                writer.add_scalar('loss/adv_D',   disc_loss.item(), global_step)
                writer.add_scalar('train/lr',     scheduler.get_last_lr()[0], global_step)

            # ── Sample preview ────────────────────────────────
            if global_step % args.sample_every == 0:
                net.eval()
                with torch.no_grad():
                    with torch.amp.autocast(device.type, enabled=USE_AMP):
                        sample = net(probe_t).squeeze(0).clamp(0, 1).cpu()
                net.train()
                sample_path = f'samples/{args.name}/step_{global_step:06d}.jpg'
                transforms.ToPILImage()(sample).save(sample_path, quality=90)
                if writer:
                    writer.add_image('sample/output', sample, global_step)
                print(f'  → sample: {sample_path}')

            if global_step % 200 == 0:
                pct = global_step / total_steps * 100
                print(f'[ep {epoch}/{args.epochs}]  '
                      f'step {global_step:>6}/{total_steps}  ({pct:5.1f}%)  '
                      f'content={c_loss.item():.4f}  style={s_loss.item():.3e}  '
                      f'stats={fs_loss.item():.3e}  tv={t_loss.item():.3e}  '
                      f'adv_G={adv_loss.item():.3e}  adv_D={disc_loss.item():.3e}  '
                      f'lr={scheduler.get_last_lr()[0]:.2e}')

            if global_step % args.checkpoint_every == 0:
                save_checkpoint(resume_path, net, disc, optim_g, optim_d,
                                scheduler, scaler, epoch, global_step, args)
                print(f'  ✓ checkpoint  (step {global_step})')

        ep_path = f'models/{args.name}_ep{epoch}.pth'
        torch.save(net.state_dict(), ep_path)
        save_checkpoint(resume_path, net, disc, optim_g, optim_d,
                        scheduler, scaler, epoch + 1, global_step, args)
        print(f'\n  → epoch {epoch} complete — {ep_path}\n')

    final = f'models/{args.name}.pth'
    torch.save({'model_state': net.state_dict(), 'n_residual': args.n_residual}, final)
    if os.path.exists(resume_path):
        os.remove(resume_path)
    if writer:
        writer.close()
    print(f'\nTraining complete.  Final model → {final}')
    if writer:
        print(f'Loss curves  : tensorboard --logdir runs')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # Required
    p.add_argument('--style',             required=True)
    p.add_argument('--dataset',           required=True)
    p.add_argument('--name',              required=True)
    # Resume / manifest
    p.add_argument('--resume',            default=None)
    p.add_argument('--manifest',          default=None,
                   help='Path to valid_images.txt from preprocess.py (faster startup)')
    # Architecture
    p.add_argument('--n-residual',        type=int,   default=9)
    # Training schedule
    p.add_argument('--epochs',            type=int,   default=4)
    p.add_argument('--batch-size',        type=int,   default=8)
    p.add_argument('--image-size',        type=int,   default=288)
    p.add_argument('--lr',                type=float, default=1e-3)
    p.add_argument('--checkpoint-every',  type=int,   default=500)
    # Loss weights
    p.add_argument('--content-weight',    type=float, default=1e5)
    p.add_argument('--style-weight',      type=float, default=4e10)
    p.add_argument('--stats-weight',      type=float, default=1e8,
                   help='Feature statistics (mean+std) loss weight')
    p.add_argument('--tv-weight',         type=float, default=1e-6)
    p.add_argument('--identity-weight',   type=float, default=1e3,
                   help='Identity loss weight (net(style)≈style)')
    p.add_argument('--identity-every',    type=int,   default=8,
                   help='Compute identity loss every N steps')
    # Per-layer style weights (relu1_2 … relu5_3)
    p.add_argument('--sw1', type=float, default=1.0,  help='Style weight relu1_2')
    p.add_argument('--sw2', type=float, default=1.0,  help='Style weight relu2_2')
    p.add_argument('--sw3', type=float, default=1.0,  help='Style weight relu3_3')
    p.add_argument('--sw4', type=float, default=1.0,  help='Style weight relu4_3')
    p.add_argument('--sw5', type=float, default=0.5,  help='Style weight relu5_3')
    p.add_argument('--adv-weight',         type=float, default=1e4,
                   help='Adversarial (PatchGAN) loss weight for the generator')
    # Monitoring
    p.add_argument('--sample-every',       type=int,   default=500,
                   help='Save a sample image every N steps (default: 500)')
    p.add_argument('--sample-image',       default=None,
                   help='Fixed probe image for samples (default: style image)')
    p.add_argument('--no-tensorboard',     action='store_true',
                   help='Disable TensorBoard logging')
    train(p.parse_args())
