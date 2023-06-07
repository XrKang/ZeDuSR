python ./downSample_isoKernel.py --image_path '../Data/WideView_GT' --save_path '../Data/WideView_iso2x' --kernel_path '../Data/kernel_WideView_iso2x'

python ./JPEG_compression.py --image_path '../Data/WideView_iso2x' --save_path '../Data/WideView_iso2x_JPEG75'

python ./center_crop.py --image_path '../Data/TeleView' --save_path '../Data/TeleView_crop'

python ./center_crop.py --image_path '../Data/WideView_iso2x_JPEG75' --save_path '../Data/WideView_iso2x_JPEG75_crop'