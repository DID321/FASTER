# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

class MaskedAutoencoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.norm_pix_loss = True
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.decoder_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.decoder_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def unpatchify_ch3(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.decoder_patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def masking_id(self, batch_size, mask_ratio):
        N, L = batch_size, self.patch_embed.num_patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=self.pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def random_masking(self, x, ids_keep):
        N, L, D = x.shape
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward_encoder(self, x, mask_ratio):
        raise NotImplementedError

    def forward_decoder(self, x, ids_restore):
        raise NotImplementedError

    def forward_loss(self, imgs, cls_pred, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        num_preds = mask.sum()
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        cls_pred, pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, cls_pred, pred, mask)
        return loss, pred, mask

    def visualize_mae_results(self, img, pred, mask):
        """
        Visualize the original image, masked image, and prediction result.

        Args:
            img (torch.Tensor): Original image tensor of shape (N, 3, H, W).
            pred (torch.Tensor): Prediction tensor of shape (N, L, patch_size**2 * 3).
            mask (torch.Tensor): Mask tensor of shape (N, L), where 0 is keep, 1 is remove.
            patch_size (int): Size of each patch.
        """
        # Unpatchify the prediction
        patch_size = self.decoder_patch_size
        pred_img = self.unpatchify_ch3(pred)


        # Mask the original image
        masked_img = img.clone()
        N, _, H, W = img.shape
        mask = mask.reshape(N, int(H / patch_size), int(W / patch_size))
        mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        # mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        mask = mask.unsqueeze(1)
        masked_img[mask == 1] = 0  # Set masked regions to black
        idx = np.random.randint(0, img.shape[0])
        # Convert tensors to numpy arrays for visualization
        img = img[idx].permute(1, 2, 0).cpu().detach().numpy()
        masked_img = masked_img[idx].permute(1, 2, 0).cpu().detach().numpy()
        pred_img0 = pred_img[idx, 0, :, :].cpu().detach().numpy()
        pred_img1 = pred_img[idx, 1, :, :].cpu().detach().numpy()
        pred_img2 = pred_img[idx, 2, :, :].cpu().detach().numpy()

        return img, masked_img, pred_img0,  pred_img1,  pred_img2
