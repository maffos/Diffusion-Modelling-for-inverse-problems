from utils import plot_csv
import os

src_dir = 'plots/losses'
for root,dirs,files in os.walk(src_dir):
    for fname in files:
        if '.csv' in fname and not 'linear' in fname:

            out_file = os.path.join(root, fname[:-4]+'.svg')
            csv_file = os.path.join(root, fname)
            print(out_file)
            plot_csv(csv_file,out_file, labelsize = 18, max_step=20000, show_plot = False)

        elif '.csv' in fname and 'linear' in fname:

            out_file = os.path.join(root, fname[:-4] + '.svg')
            csv_file = os.path.join(root, fname)
            print(out_file)
            plot_csv(csv_file, out_file, labelsize = 18, max_step=1600, show_plot = False)

