import numpy as np
import matplotlib.pyplot as plt

def active_semi_sumpervised_learning(model, unlabeled_ds, threshold=0.9, batch_size=32):
    pseudo_X, pseudo_y = [], []
    manual_X, manual_y = [], []

    for images, filenames in unlabeled_ds:
        preds = model.predict(images, verbose=0)
        conf = np.max(preds, axis=1)
        labels = np.argmax(preds, axis=1)

        mask_confident = conf >= threshold
        mask_uncertain = ~mask_confident

        if np.any(mask_confident):
            pseudo_X.append(images[mask_confident])
            pseudo_y.append(labels[mask_confident])

        if np.any(mask_uncertain):
            for img, fname in zip(images[mask_uncertain], filenames[mask_uncertain]):
                plt.imshow((img + 1) / 2)
                plt.title(fname.decode("utf-8") if isinstance(fname, bytes) else fname)
                plt.axis('off')
                plt.show()

                correct_label = int(input(f"Enter correct label for {fname}: "))
                manual_X.append(img)
                manual_y.append(correct_label)

    if pseudo_X:
        pseudo_X = np.concatenate(pseudo_X, axis=0)
        pseudo_y = np.concatenate(pseudo_y, axis=0)
    else:
        pseudo_X = np.empty((0, 224,224,3), dtype=np.float32)
        pseudo_y = np.empty((0,), dtype=np.int32)

    if manual_X:
        manual_X = np.stack(manual_X)
        manual_y = np.array(manual_y, dtype=np.int32)
    else:
        manual_X = np.empty((0,224,224,3), dtype=np.float32)
        manual_y = np.empty((0,), dtype=np.int32)

    return pseudo_X, pseudo_y, manual_X, manual_y