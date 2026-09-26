import os; os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse, cv2, torch
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument('--model', default='train/weights/best.pt')
p.add_argument('--source', default='media/test.mkv')
p.add_argument('--stream', default='store_true')
p.add_argument('--device', default=None)
p.add_argument('--imgsz', type=int, default=640)
p.add_argument('--conf', type=float, default=0.25)
a = p.parse_args()

a.device = a.device or ('mps' if torch.backends.mps.is_available() else '0' if torch.cuda.is_available() else 'cpu')
src = a.source.lstrip('usb') if a.source.startswith('usb') else a.source
is_img = src.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
if not is_img and not a.stream: a.stream = input('Stream preview? [y/N]: ').strip().lower() == 'y'

os.makedirs('output', exist_ok=True)
fps = 30.0
if not is_img: c = cv2.VideoCapture(src); fps = c.get(cv2.CAP_PROP_FPS) or 30.0; c.release()
writer, frames = None, 0

for r in YOLO(a.model).predict(source=src, stream=True, device=a.device, half=True, 
                               imgsz=a.imgsz, conf=a.conf, verbose=False):
    img = r.plot()

    if is_img:
        cv2.imwrite('output/result.jpg', img=img); print("Saved -> output/result.jpg"); break

    if not writer: writer = cv2.VideoWriter('output/result.mp4', 
                                            cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                                            (img.shape[1], img.shape[0]))
    writer.write(img); frames += 1

    if a.stream:
        cv2.imshow('Detection', img)
        if cv2.waitKey(1) == ord('q'): break
    elif frames % 30 == 0: print(f'frames: {frames}', end='\r')

if writer: writer.release(); print(f'\nVideo saved -> output/result.mp4 ({frames} frames)')
cv2.destroyAllWindows()