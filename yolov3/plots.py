# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Export a  PyTorch model to TorchScript, ONNX, CoreML, TensorFlow (saved_model, pb, TFLite, TF.js,) formats
TensorFlow exports authored by https://github.com/zldrobit

Usage:
    $ python path/to/export.py --weights yolov3.pt --include torchscript onnx coreml saved_model pb tflite tfjs

Inference:
    $ python path/to/detect.py --weights yolov3.pt
                                         yolov3.onnx  (must export with --dynamic)
                                         yolov3_saved_model
                                         yolov3.pb
                                         yolov3.tflite

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov3_web_model public/yolov3_web_model
    $ npm start
"""

import matplotlib.pyplot as plt
import pandas as pd
res = pd.read_csv('runs/train/exp/results.csv')
res['train_loss'] = res[ '      train/obj_loss']+ res['      train/box_loss']+res['      train/cls_loss']
res['val_loss'] = res['        val/box_loss'] + res['        val/cls_loss']+res['        val/obj_loss']

fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
ax = ax.ravel()

data = res.copy()
s = [x.strip() for x in data.columns]
x = data.values[:, 0]
for i, j in enumerate([1, 2, 3, 16, 10, 11, 12, 17]):
    y = data.values[:, j]
    # y[y == 0] = np.nan  # don't show zero values
    ax[i].plot(x, y, marker='.', linewidth=2, markersize=8)
    ax[i].set_title(s[j], fontsize=12)
    # if j in [8, 9, 10]:  # share train and val loss y axes
    #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

ax[1].legend()
# fig.savefig(save_dir / 'results.png', dpi=200)
# plt.close()
# plt.show()
fig.savefig('runs/train/exp/results_loss.png', dpi=200)



