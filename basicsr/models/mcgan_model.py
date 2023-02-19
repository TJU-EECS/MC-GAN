import importlib
import torch
from collections import OrderedDict
from copy import deepcopy

from basicsr.models.archs import define_network
from basicsr.models.sr_model import SRModel

loss_module = importlib.import_module('basicsr.models.losses')


def padding(data, fill_scale, mode='reflect'):
    _, _, H, W = data.shape
    pad_x = (H + fill_scale - 1) // fill_scale * fill_scale - H
    pad_y = (W + fill_scale - 1) // fill_scale * fill_scale - W
    out_data = torch.nn.functional.pad(data, pad=[0, pad_y, 0, pad_x], mode=mode, value=0)
    return out_data, H, W


def de_padding(hr, scale, H, W):
    hr = hr[:, :, 0: H * scale, 0: W * scale]
    return hr


def reduce_image(img, scale):
    batch, channels, height, width = img.size()
    reduced_img = torch.zeros(batch, channels * scale * scale, height // scale, width // scale).cuda()

    for x in range(scale):
        for y in range(scale):
            for c in range(channels):
                reduced_img[:, c + channels * (y + scale * x), :, :] = img[:, c, x::scale, y::scale]
    return reduced_img


def reconstruct_image(features, scale):
    batch, channels, height, width = features.size()
    img_channels = channels // (scale**2)
    reconstructed_img = torch.zeros(batch, img_channels, height * scale, width * scale).cuda()

    for x in range(scale):
        for y in range(scale):
            for c in range(img_channels):
                f_channel = c + img_channels * (y + scale * x)
                reconstructed_img[:, c, x::scale, y::scale] = features[:, f_channel, :, :]
    return reconstructed_img


def patchify_tensor(features, patch_size, overlap=10):
    batch_size, channels, height, width = features.size()

    effective_patch_size = patch_size - overlap
    n_patches_height = (height // effective_patch_size)
    n_patches_width = (width // effective_patch_size)

    if n_patches_height * effective_patch_size < height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < width:
        n_patches_width += 1

    patches = []
    for b in range(batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, height - patch_size)
                patch_start_width = min(w * effective_patch_size, width - patch_size)
                patches.append(features[b:b+1, :,
                               patch_start_height: patch_start_height + patch_size,
                               patch_start_width: patch_start_width + patch_size])
    return torch.cat(patches, 0)


def recompose_tensor(patches, full_height, full_width, overlap=10):

    batch_size, channels, patch_size, _ = patches.size()
    effective_patch_size = patch_size - overlap
    n_patches_height = (full_height // effective_patch_size)
    n_patches_width = (full_width // effective_patch_size)

    if n_patches_height * effective_patch_size < full_height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < full_width:
        n_patches_width += 1

    n_patches = n_patches_height * n_patches_width
    if batch_size % n_patches != 0:
        print("Error: The number of patches provided to the recompose function does not match the number of patches in each image.")
    final_batch_size = batch_size // n_patches

    blending_in = torch.linspace(0.1, 1.0, overlap)
    blending_out = torch.linspace(1.0, 0.1, overlap)
    middle_part = torch.ones(patch_size - 2 * overlap)
    blending_profile = torch.cat([blending_in, middle_part, blending_out], 0)

    horizontal_blending = blending_profile[None].repeat(patch_size, 1)
    vertical_blending = blending_profile[:, None].repeat(1, patch_size)
    blending_patch = horizontal_blending * vertical_blending

    blending_image = torch.zeros(1, channels, full_height, full_width)
    for h in range(n_patches_height):
        for w in range(n_patches_width):
            patch_start_height = min(h * effective_patch_size, full_height - patch_size)
            patch_start_width = min(w * effective_patch_size, full_width - patch_size)
            blending_image[0, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += blending_patch[None]

    recomposed_tensor = torch.zeros(final_batch_size, channels, full_height, full_width)
    if patches.is_cuda:
        blending_patch = blending_patch.cuda()
        blending_image = blending_image.cuda()
        recomposed_tensor = recomposed_tensor.cuda()
    patch_index = 0
    for b in range(final_batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, full_height - patch_size)
                patch_start_width = min(w * effective_patch_size, full_width - patch_size)
                recomposed_tensor[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches[patch_index] * blending_patch
                patch_index += 1
    recomposed_tensor /= blending_image

    return recomposed_tensor


class mcModel(SRModel):
    

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network net_d
        self.net_d = define_network(deepcopy(self.opt['network_d']))
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path'].get('strict_load_d', True))

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)

        if train_opt.get('KLdiv_opt'):
            self.kl_weight = train_opt['KLdiv_opt']['loss_weight']
            if self.kl_weight != 0:
                self.cri_kldiv = True
            else:
                self.cri_kldiv = False
        else:
            self.cri_kldiv = None

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(),
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                **train_opt['optim_d'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output, mu, logvar = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0
                and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(
                    self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # kldiv loss
            if self.cri_kldiv:
                l_g_kl = torch.mean(-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp(), dim=1), dim=0) * self.kl_weight
                l_g_total += l_g_kl
                loss_dict['l_g_kl'] = l_g_kl
            # gan loss
            fake_g_pred_style, fake_g_pred_defect = self.net_d(self.output)
            l_g_gan_style = self.cri_gan(fake_g_pred_style, True, is_disc=False)
            l_g_gan_defect = self.cri_gan(fake_g_pred_defect, True, is_disc=False)
            l_g_total += l_g_gan_style
            l_g_total += l_g_gan_defect
            loss_dict['l_g_gan'] = l_g_gan_style + l_g_gan_defect

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred_style, real_d_pred_defect = self.net_d(self.gt)
        l_d_real_style = self.cri_gan(real_d_pred_style, True, is_disc=True)
        l_d_real_defect = self.cri_gan(real_d_pred_defect, True, is_disc=True)
        l_d_real = l_d_real_style + l_d_real_defect
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred_style.detach()) + torch.mean(real_d_pred_defect.detach())
        l_d_real.backward()
        # fake
        fake_d_pred_style, fake_d_pred_defect = self.net_d(self.output.detach())
        l_d_fake_style = self.cri_gan(fake_d_pred_style, False, is_disc=True)
        l_d_fake_defect = self.cri_gan(fake_d_pred_defect, False, is_disc=True)
        l_d_fake = l_d_fake_style + l_d_fake_defect
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred_style.detach()) + torch.mean(fake_d_pred_defect.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        valid_opt = self.opt['val']
        overlap = valid_opt['overlap']
        val_scale = valid_opt['val_scale']
        valid_patch = valid_opt['valid_patch']
        self.net_g.eval()
        with torch.no_grad():
            out_box = []
            pad_lq, ori_H, ori_W = padding(data=self.lq, fill_scale=valid_patch)
            b, c, h, w = pad_lq.size()
            lowres_patches = patchify_tensor(pad_lq, patch_size=valid_patch, overlap=overlap)
            for p in range(lowres_patches.size(0)):
                hr, _, _ = self.net_g(lowres_patches[p:p + 1])
                out_box.append(hr)
            out_box = torch.cat(out_box, dim=0)
            hr_image = recompose_tensor(out_box, val_scale * h, val_scale * w, overlap=val_scale * overlap)
            self.output = de_padding(hr_image, val_scale, ori_H, ori_W)
            # self.output, _, _ = self.net_g(self.lq)
        self.net_g.train()

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
