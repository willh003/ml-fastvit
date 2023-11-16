import torch
import numpy as np

"""
Measure performance of different models
"""

def define_fastervit_model():
    from faster_vit import faster_vit_factory
    MODEL_PATH="/home/pcgta/Documents/playground/ml-fastvit/fastervit_checkpoints/fastervit_0_224_1k.pth.tar"
    model = faster_vit_factory(pretrained=True, model_path = MODEL_PATH)
    return model

model = define_fastervit_model()

x = torch.rand(1, 3, 224, 224)
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
        x = torch.rand(1, 3, 224, 224)

        starter.record()
        _ = model(x)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(f'{1000 / mean_syn} hz' )