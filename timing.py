import torch
import torchvision
import numpy as np
model = torchvision.models.detection.maskrcnn_resnet50_fpn().cuda()
model.eval()
x = [torch.rand(3, 300, 400)]

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 30
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = model(x)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        print(f'rep {rep}')
        x = [torch.rand(3, 300, 400)]

        starter.record()
        _ = model(x)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)