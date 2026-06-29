from decord import VideoReader, cpu

video_path = r"F:/HolmesVAU/HolmesVAU-master/examples/test_01_fixed.mp4"

vr = VideoReader(video_path, ctx=cpu(0))
print("len =", len(vr))

frame0 = vr[0]
print("frame0 shape =", frame0.shape)