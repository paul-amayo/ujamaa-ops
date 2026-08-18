"""Free VRAM in whole GB. A FILE, not `python -c`: pixi's deno_task_shell eats
the quoting in `-c` and the command silently returns nothing, which turns a
VRAM safety gate into a no-op."""
import torch
print(int(torch.cuda.mem_get_info()[0] / 1e9))
