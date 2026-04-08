import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path

# --- 全局配置 ---
PROJECT_ROOT = Path(__file__).parent
# 数据集路径：指向包含CelebA图片的内层img_align_celeba目录
DATASET_PATH = PROJECT_ROOT / 'data' / 'img_align_celeba' / 'img_align_celeba'
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # 自动创建结果目录

# 超参数（CPU环境建议调小以加速训练）
LATENT_DIM = 100
BATCH_SIZE = 32  # CPU环境从128降至32，减少计算压力
EPOCHS = 5       # CPU环境从50降至5，先验证逻辑
LEARNING_RATE = 0.0002

# 设备配置：自动检测GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# --- 数据预处理 ---
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 检查数据集路径有效性
if not DATASET_PATH.is_dir():
    raise FileNotFoundError(f"数据集路径不存在: {DATASET_PATH}\n请确认CelebA图片文件存放在该目录下")

# 自定义数据集加载类（适配CelebA直接存放图片的结构）
class SimpleImageFolder(datasets.VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # 筛选所有图片格式文件
        self.image_paths = sorted([
            p for p in Path(root).iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ])
        if not self.image_paths:
            raise FileNotFoundError(f"指定目录下未找到图片文件: {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = datasets.folder.default_loader(img_path)  # 通用图片加载器
        if self.transform is not None:
            image = self.transform(image)
        return image, 0  # 返回dummy label（GAN/VAE无需标签）

# 加载数据集（根据设备自动适配pin_memory）
dataset = SimpleImageFolder(root=DATASET_PATH, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Windows下num_workers设为0避免多进程报错
    pin_memory=True if device.type == 'cuda' else False  # 仅GPU启用pin_memory
)

# --- GAN 模型定义 ---
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

# --- VAE 模型定义 ---
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- 训练函数 ---
def train_gan(generator, discriminator, dataloader, epochs, latent_dim, device):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    real_label = 1.
    fake_label = 0.

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # 训练判别器
            optimizer_d.zero_grad()
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_imgs)
            d_loss_real = criterion(output, label)
            d_loss_real.backward()

            # 生成假图片并训练判别器
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(output, label)
            d_loss_fake.backward()
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake_imgs)
            g_loss = criterion(output, label)
            g_loss.backward()
            optimizer_g.step()

            # 每100步打印一次损失
            if (i + 1) % 100 == 0:
                print(f'GAN 轮次 [{epoch + 1}/{epochs}], 步数 [{i + 1}/{len(dataloader)}], '
                      f'判别器损失: {d_loss.item():.4f}, 生成器损失: {g_loss.item():.4f}')

        # 每轮结束保存生成的图片
        with torch.no_grad():
            fake_imgs = generator(torch.randn(16, latent_dim, 1, 1, device=device))
            save_image(fake_imgs, RESULTS_DIR / f'gan_epoch_{epoch + 1}.png', normalize=True)

def train_vae(vae, dataloader, epochs, device):
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        total_loss = 0
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            optimizer.zero_grad()

            # 前向传播
            recon_imgs, mu, logvar = vae(real_imgs)
            # 计算损失（重构损失 + KL散度）
            recon_loss = nn.functional.mse_loss(recon_imgs, real_imgs, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss

            # 反向传播与优化
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 每100步打印一次损失
            if (i + 1) % 100 == 0:
                print(f'VAE 轮次 [{epoch + 1}/{epochs}], 步数 [{i + 1}/{len(dataloader)}], '
                      f'单批次损失: {loss.item() / real_imgs.size(0):.4f}')

        # 计算本轮平均损失
        avg_loss = total_loss / len(dataset)
        print(f'VAE 轮次 [{epoch + 1}/{epochs}], 平均损失: {avg_loss:.4f}')

        # 每轮结束保存生成的图片
        with torch.no_grad():
            z = torch.randn(16, LATENT_DIM, device=device)
            generated_imgs = vae.decode(z)
            save_image(generated_imgs, RESULTS_DIR / f'vae_epoch_{epoch + 1}.png', normalize=True)

# --- 评估函数 ---
def evaluate_models(gan_gen, vae_model, dataloader, device, num_samples=5):
    """
    同时评估GAN和VAE，生成定量指标（PSNR/SSIM）和定性对比图
    """
    gan_gen.eval()
    vae_model.eval()

    psnr_gan_list, ssim_gan_list = [], []
    psnr_vae_list, ssim_vae_list = [], []

    # 获取一批真实图片
    real_imgs, _ = next(iter(dataloader))
    real_imgs = real_imgs[:num_samples].to(device)

    with torch.no_grad():
        # GAN生成图片
        gan_noise = torch.randn(num_samples, LATENT_DIM, 1, 1, device=device)
        gan_imgs = gan_gen(gan_noise)

        # VAE重建图片
        vae_recon_imgs, _, _ = vae_model(real_imgs)

        # 反归一化（将张量从[-1,1]转回[0,1]并转为numpy）
        def denorm(tensor):
            return tensor.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5

        real_np = denorm(real_imgs)
        gan_np = denorm(gan_imgs)
        vae_np = denorm(vae_recon_imgs)

        # 计算每张图片的PSNR和SSIM
        for i in range(num_samples):
            # GAN指标
            psnr_gan = peak_signal_noise_ratio(real_np[i], gan_np[i], data_range=1.0)
            ssim_gan = structural_similarity(real_np[i], gan_np[i], channel_axis=2, data_range=1.0)
            psnr_gan_list.append(psnr_gan)
            ssim_gan_list.append(ssim_gan)

            # VAE指标
            psnr_vae = peak_signal_noise_ratio(real_np[i], vae_np[i], data_range=1.0)
            ssim_vae = structural_similarity(real_np[i], vae_np[i], channel_axis=2, data_range=1.0)
            psnr_vae_list.append(psnr_vae)
            ssim_vae_list.append(ssim_vae)

        # 生成定性对比图
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # 真实图片
            axes[i, 0].imshow(real_np[i].clip(0, 1))  # 防止数值溢出
            axes[i, 0].set_title('真实图片 (GT)')
            axes[i, 0].axis('off')

            # GAN生成图片
            axes[i, 1].imshow(gan_np[i].clip(0, 1))
            axes[i, 1].set_title(f'GAN生成\nPSNR: {psnr_gan_list[i]:.2f}, SSIM: {ssim_gan_list[i]:.2f}')
            axes[i, 1].axis('off')

            # VAE重建图片
            axes[i, 2].imshow(vae_np[i].clip(0, 1))
            axes[i, 2].set_title(f'VAE重建\nPSNR: {psnr_vae_list[i]:.2f}, SSIM: {ssim_vae_list[i]:.2f}')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / '生成图片对比图.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 计算平均指标
    avg_gan_psnr = sum(psnr_gan_list) / len(psnr_gan_list)
    avg_gan_ssim = sum(ssim_gan_list) / len(ssim_gan_list)
    avg_vae_psnr = sum(psnr_vae_list) / len(psnr_vae_list)
    avg_vae_ssim = sum(ssim_vae_list) / len(ssim_vae_list)

    return (avg_gan_psnr, avg_gan_ssim), (avg_vae_psnr, avg_vae_ssim)

# --- 主函数（执行训练和评估） ---
if __name__ == "__main__":
    # 初始化模型
    generator_gan = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    vae_model = VAE(LATENT_DIM).to(device)

    # 训练GAN
    print("\n=== 开始训练GAN模型 ===")
    train_gan(generator_gan, discriminator, dataloader, EPOCHS, LATENT_DIM, device)

    # 训练VAE
    print("\n=== 开始训练VAE模型 ===")
    train_vae(vae_model, dataloader, EPOCHS, device)

    # 评估模型
    print("\n=== 开始评估模型性能 ===")
    (gan_psnr, gan_ssim), (vae_psnr, vae_ssim) = evaluate_models(generator_gan, vae_model, dataloader, device)

    # 输出定量评估结果
    print("\n==================== 定量评估结果 ====================")
    print(f"GAN模型 - 平均PSNR: {gan_psnr:.4f}, 平均SSIM: {gan_ssim:.4f}")
    print(f"VAE模型 - 平均PSNR: {vae_psnr:.4f}, 平均SSIM: {vae_ssim:.4f}")

    # 输出结论
    print("\n==================== 实验结论 ====================")
    if gan_psnr > vae_psnr and gan_ssim > vae_ssim:
        print("结论：在PSNR和SSIM两个核心指标上，GAN模型表现均优于VAE模型，说明GAN生成的图像在像素层面与真实图像更相似。")
    elif vae_psnr > gan_psnr and vae_ssim > gan_ssim:
        print("结论：在PSNR和SSIM两个核心指标上，VAE模型表现均优于GAN模型，说明VAE的图像重建质量更高。")
    else:
        print("结论：GAN和VAE在定量指标上各有优劣——GAN生成的图像多样性更强，但像素级相似度略低；VAE重建图像更稳定，但多样性不足。")
    print(f"\n所有生成的图片（训练过程+对比图）已保存至：{RESULTS_DIR}")