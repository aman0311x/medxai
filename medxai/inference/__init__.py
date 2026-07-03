import torch


def sliding_window_inference_2d(model, image, patch_size=(64, 64), stride=(32, 32)):
    """
    Performs patch-based sliding window inference on a large 2D image tensor to prevent OOM errors.
    """
    model.eval()
    _, C, H, W = image.shape
    ph, pw = patch_size
    sh, sw = stride

    output = torch.zeros((1, model.n_classes, H, W), device=image.device)
    importance_weight = torch.zeros((1, model.n_classes, H, W), device=image.device)

    with torch.no_grad():
        for y in range(0, H - ph + 1, sh):
            for x in range(0, W - pw + 1, sw):
                patch = image[:, :, y : y + ph, x : x + pw]
                patch_out = model(patch)

                output[:, :, y : y + ph, x : x + pw] += patch_out
                importance_weight[:, :, y : y + ph, x : x + pw] += 1.0

    importance_weight[importance_weight == 0] = 1.0
    return output / importance_weight
