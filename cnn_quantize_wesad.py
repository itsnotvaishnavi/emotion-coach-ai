import os
import time
import torch
import torch.nn as nn

from cnn_train_wesad import SimpleCNN


def get_file_size_kb(path):
    return round(os.path.getsize(path) / 1024, 2)


@torch.no_grad()
def measure_inference_time(model, device="cpu", runs=200):
    model.eval()

    # dummy input: (batch=1, channels=6, seq_len=320)
    x = torch.randn(1, 6, 320)

    if device == "cpu":
        model.to("cpu")
        x = x.to("cpu")
    else:
        model.to(device)
        x = x.to(device)

    # warmup
    for _ in range(20):
        _ = model(x)

    start = time.time()
    for _ in range(runs):
        _ = model(x)
    end = time.time()

    avg_ms = ((end - start) / runs) * 1000
    return avg_ms


if __name__ == "__main__":

    # Load FP32 model
    fp32_model = SimpleCNN(in_channels=6, num_classes=4)
    fp32_model.load_state_dict(torch.load("cnn_wesad.pth", map_location="cpu"))
    fp32_model.eval()

    # Save fp32 size
    fp32_path = "cnn_fp32.pth"
    torch.save(fp32_model.state_dict(), fp32_path)

    print("FP32 model size (KB):", get_file_size_kb(fp32_path))

    # Dynamic Quantization (works mainly on Linear layers)
    quant_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8
    )
    quant_model.eval()

    # Save quantized model
    quant_path = "cnn_int8_dynamic.pth"
    torch.save(quant_model.state_dict(), quant_path)

    print("INT8 dynamic model size (KB):", get_file_size_kb(quant_path))

    # Inference time (CPU)
    fp32_time = measure_inference_time(fp32_model, device="cpu", runs=200)
    int8_time = measure_inference_time(quant_model, device="cpu", runs=200)

    print("\n==============================")
    print("INFERENCE TIME (CPU)")
    print("==============================")
    print("FP32 avg ms:", round(fp32_time, 4))
    print("INT8 avg ms:", round(int8_time, 4))
