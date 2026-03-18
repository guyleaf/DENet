import torch

from denet.engine.common import evaluate, init_model
from denet.engine.options import TestOptions
from denet.data.datasets import create_dataloader
from denet.utils.yolo_utils import check_img_size


def _torch_load(f, map_location=None):
    """torch.load with weights_only=False for PyTorch >= 1.13, plain call for older."""
    try:
        return torch.load(f, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(f, map_location=map_location)


def test(opt, model, data_dict):

    # get config
    batch_size, img_size_test, checkpoint, device = opt.batch_size, opt.img_size_test, opt.checkpoint, opt.device

    # load state_dict
    if checkpoint.strip() != '':
        ckpt = _torch_load(checkpoint, map_location=device)  # load checkpoint

        # model state_dict
        model._load_state_dict_(ckpt['model'])

    img_size_test = check_img_size(img_size_test, s=model.Detect.stride.max())

    # test dataloader
    dataloader = create_dataloader(
        data_dict=data_dict,
        img_size=img_size_test,
        batch_size=batch_size,
        task=opt.task,
    )

    print(f'test on image sizes {img_size_test} test')
    is_coco = data_dict.get('is_coco', False)

    evaluate(
        model=model,
        dataloader=dataloader,
        batch_size=batch_size,
        device=device,
        tb_writer=opt.tb_writer,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        run_dir=opt.run_dir,
        verbose=opt.verbose,
        save_json=True,
        is_coco=is_coco,
    )


if __name__ == '__main__':

    opt = TestOptions().init()  # get test options

    model, data_dict, hyp = init_model(opt)  # init model

    test(opt, model, data_dict)
