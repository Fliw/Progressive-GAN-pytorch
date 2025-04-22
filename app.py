import torch, os, gradio as gr, numpy as np
from torchvision import utils, transforms
from progan_modules import Generator

CHECKPOINT_DIR = "./model"
Z_DIM, CHANNEL_SIZE = 128, 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIXED_STEP  = 6
FIXED_ALPHA = 0.0

g_running = Generator(CHANNEL_SIZE, Z_DIM, pixel_norm=False, tanh=False).to(DEVICE)
g_running.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "g.model"), map_location=DEVICE))
g_running.eval()

to_pil = transforms.ToPILImage()

@torch.inference_mode()
def sample_images(n_images: int = 50, seed: int | None = None):
    if seed is not None and seed >= 0:
        torch.manual_seed(seed); np.random.seed(seed)
    else:
        torch.seed()

    z = torch.randn(n_images, Z_DIM, device=DEVICE)
    imgs = g_running(z, step=FIXED_STEP, alpha=FIXED_ALPHA).cpu()

    grid = utils.make_grid(imgs, nrow=10, normalize=True, value_range=(-1, 1))
    return to_pil(grid)

demo = gr.Interface(
    fn=sample_images,
    inputs=[
        gr.Slider(1, 200, value=50, step=10, label="Jumlah Gambar (kelipatan 10)"),
        gr.Number(value=-1, precision=0, label="Seed (‑1 = acak)"),
    ],
    outputs=gr.Image(type="pil", label="Grid Hasil"),
    title="Progressive Growing Generative Adversarial Network",
    description="contoh implementasi PGGAN untuk dataset jerawat",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.queue()
    demo.launch(show_api=False, share=True)
