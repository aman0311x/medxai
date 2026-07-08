import torch


def sliding_window_inference_2d(model, image, patch_size=(64, 64), stride=(32, 32)):
    """
    Performs patch-based sliding window inference on a large 2D image tensor to prevent OOM errors.
    """
    model.eval()
    _, C, H, W = image.shape
    ph, pw = patch_size
    sh, sw = stride

    if H < ph or W < pw:
        raise ValueError(
            f"Image size ({H}, {W}) is smaller than patch_size ({ph}, {pw}). "
            "Sliding window inference requires image dimensions >= patch size."
        )

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


def sliding_window_inference_3d(
    model, volume, patch_size=(64, 64, 64), stride=(32, 32, 32)
):
    """
    Performs patch-based sliding window inference on a large 3D volume
    tensor (e.g. CT/MRI scan) to prevent OOM errors. Mirrors
    sliding_window_inference_2d for volumetric UNet3D-style models.
    """
    model.eval()
    _, C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    if D < pd or H < ph or W < pw:
        raise ValueError(
            f"Volume size ({D}, {H}, {W}) is smaller than patch_size "
            f"({pd}, {ph}, {pw}). Sliding window inference requires "
            "volume dimensions >= patch size."
        )

    output = torch.zeros((1, model.n_classes, D, H, W), device=volume.device)
    importance_weight = torch.zeros((1, model.n_classes, D, H, W), device=volume.device)

    with torch.no_grad():
        for z in range(0, D - pd + 1, sd):
            for y in range(0, H - ph + 1, sh):
                for x in range(0, W - pw + 1, sw):
                    patch = volume[:, :, z : z + pd, y : y + ph, x : x + pw]
                    patch_out = model(patch)

                    output[:, :, z : z + pd, y : y + ph, x : x + pw] += patch_out
                    importance_weight[:, :, z : z + pd, y : y + ph, x : x + pw] += 1.0

    importance_weight[importance_weight == 0] = 1.0
    return output / importance_weight
