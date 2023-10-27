import torch
import torchvision
import numpy as np

# from vit_semantic import VitSemantic
# from models.fastvit import fastvit_ma36
# backbone = fastvit_ma36(fork_feat=True)
# checkpoint = torch.load('/home/pcgta/Documents/playground/ml-fastvit/pretrained_checkpoints/fastvit_ma36.pth.tar')
# backbone.load_state_dict(checkpoint['state_dict'])
# model = VitSemantic(2, (224,224), backbone=backbone)


from faster_vit import create_faster_vit
MODEL_PATH="/home/pcgta/Documents/playground/ml-fastvit/pretrained_checkpoints/fastervit_0_224_1k.pth.tar"
model = create_faster_vit(pretrained=True, model_path = MODEL_PATH)



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