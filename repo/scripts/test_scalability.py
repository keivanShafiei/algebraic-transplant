import math
import time
import torch
from src.projection.layer import SparseHelmholtzProjection
from src.rbf_fd.operators import assemble_divergence_operator

def make_rect_grid(nx: int, ny: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, device=device)
    y = torch.linspace(0.0, 1.0, ny, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

def make_local_stencils(nx: int, ny: int, device: torch.device, radius: int = 2) -> torch.Tensor:
    offsets =[(di, dj) for di in range(-radius, radius + 1) for dj in range(-radius, radius + 1)]
    ii = torch.arange(nx, device=device).view(nx, 1).expand(nx, ny)
    jj = torch.arange(ny, device=device).view(1, ny).expand(nx, ny)
    stencil_list =[]
    for di, dj in offsets:
        ni = (ii + di).clamp(0, nx - 1)
        nj = (jj + dj).clamp(0, ny - 1)
        stencil_list.append((ni * ny + nj).reshape(-1))
    return torch.stack(stencil_list, dim=1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    nx, ny = 250, 400
    N = nx * ny

    points = make_rect_grid(nx, ny, device)
    stencils = make_local_stencils(nx, ny, device, radius=2)

    h = math.sqrt(1.0 / N)
    c = 1.2 * h

    t0 = time.perf_counter()
    G = assemble_divergence_operator(points, stencils, c, sparse=True)
    proj = SparseHelmholtzProjection(G, tol=1e-5, max_iter=1500).to(device)
    
    a_hat = torch.randn(G.shape[1], 1, device=device, dtype=torch.float32)
    
    # Pre-projection divergence
    div_initial = torch.linalg.norm(torch.sparse.mm(G, a_hat)).item()
    
    a_NO, q = proj(a_hat, return_q=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    residual = torch.linalg.norm(torch.sparse.mm(G, a_NO)).item()
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0

    rel_res = residual / (div_initial + 1e-12)

    print(f"N                : {N}")
    print(f"G shape          : {tuple(G.shape)}")
    print(f"nnz              : {G._nnz()}")
    print(f"execution time   : {t1 - t0:.3f} s")
    print(f"peak VRAM        : {peak_vram_gb:.3f} GB")
    print(f"Initial ||G a||  : {div_initial:.3e}")
    print(f"Final ||G a||    : {residual:.3e}")
    print(f"Relative res     : {rel_res:.3e}")

    assert rel_res < 5e-3, f"Relative projection residual too large: {rel_res:.3e}"
    print("✅ Scalability test passed successfully!")

if __name__ == "__main__":
    main()
