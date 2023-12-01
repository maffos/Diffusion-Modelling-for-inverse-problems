from models.diffusion import *
from examples.scatterometry.main_diffusion import evaluate
from examples.scatterometry.utils_scatterometry import get_forward_model_params,get_dataset
import os
import shutil

if __name__ == '__main__':

    surrogate_dir = 'examples/scatterometry'
    src_dir = 'examples/scatterometry/results'
    gt_dir = 'examples/scatterometry/gt_samples'

    forward_model,a,b,lambd_bd,xdim,ydim = get_forward_model_params(surrogate_dir)
    n_samples_x = 30000
    n_samples_y = 100
    x_test,y_test = get_dataset(forward_model,a,b,size = n_samples_y)
    for root, dirs, files in os.walk(src_dir):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            chkpnt_path = os.path.join(subfolder_path, 'diffusion.pt')
            if os.path.isfile(chkpnt_path):
                out_dir = os.path.join(subfolder_path, 'results2')
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                model = create_diffusion_model2(xdim, ydim, hidden_layers=[512, 512, 512])
                checkpoint = torch.load(chkpnt_path, map_location=torch.device(device))
                model.a.load_state_dict(checkpoint)
                evaluate(model, y_test[:100], forward_model,a,b,lambd_bd,out_dir, gt_dir, n_samples_x=n_samples_x, n_repeats=10)