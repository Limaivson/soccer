import os
import argparse
import cv2
import torch
from ultralytics import YOLO
from custom_markers import draw_annotations, jersey_colors, draw_possession
from team_tracker import TeamTracker
from possession_tracker import PossessionTracker


if torch.cuda.is_available():
    DEVICE = "cuda:0"

    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
else:
    DEVICE = "cpu"
    print("CUDA não disponível. Usando CPU.")

p = argparse.ArgumentParser(
    description="YOLO + Team Tracker"
)

p.add_argument("--model", default="train/weights/best.pt")
p.add_argument("--source", default="media/test.mp4")
p.add_argument("--stream", action="store_true", help="Mostra o vídeo em tempo real")
p.add_argument("--device", default=None, help="Ex: cuda:0, cuda:1, cpu")
p.add_argument("--imgsz", type=int, default=1024)
p.add_argument("--conf", type=float, default=0.65)
p.add_argument("--half", action="store_true", help="Usa FP16 na GPU")
p.add_argument("--vid-stride", type=int, default=1, help="Processa 1 a cada N frames")
p.add_argument("--save", action="store_true", help="Salva o vídeo processado", default=True)
a = p.parse_args()

device = a.device or DEVICE

if device.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA solicitada, mas não está disponível.")
    print("Usando CPU.")
    device = "cpu"


half = a.half and device.startswith("cuda")

src = a.source

if src.startswith("usb"):
    src = src[3:]

is_img = src.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))

is_video = not is_img

if is_video and not a.stream:
    a.stream = (
        input("Stream preview? [y/N]: ")
        .strip()
        .lower()
        == "y"
    )

os.makedirs("output", exist_ok=True)

fps = 30.0
width = None
height = None

if is_video:

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

print("\nCarregando modelo...")

model = YOLO(a.model)

# Faz um warmup da GPU
if device.startswith("cuda"):
    model.to(device)

    dummy = torch.zeros(1, 3, a.imgsz, a.imgsz, device=device)

    with torch.inference_mode():
        model.model(dummy)

    torch.cuda.synchronize()
    print("GPU warmup concluído.")


print(f"Device : {device}")
print(f"Image  : {a.imgsz}")
print(f"FP16   : {half}")
print(f"Conf   : {a.conf}")
print(f"Stride : {a.vid_stride}")

tracker, pos_tracker = TeamTracker(), PossessionTracker()

writer = None
frames = 0

with torch.inference_mode():

    results = model.predict(
        source=src,
        stream=True,
        device=device,
        imgsz=a.imgsz,
        conf=a.conf,
        half=half,
        vid_stride=a.vid_stride,
        verbose=False,
        save=False,
        show=False,
    )

    for r in results:

        img = r.orig_img

        if r.boxes is not None and len(r.boxes) > 0:

            boxes = r.boxes

            b = boxes.xyxy.cpu().numpy()
            c = boxes.cls.cpu().numpy()

            names = r.names

            colors = [
                jersey_colors(
                    img,
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
                for x1, y1, x2, y2 in b
            ]

            team_ids, team_colors = tracker.assign(colors)

            ball = next((x for x, cid in zip(b, c) if 'ball' in names[int(cid)].lower()), None)
            pos_tracker.update(ball, b, team_ids)

            draw_annotations(img, b, c, names, team_ids, team_colors)
            draw_possession(img, pos_tracker.percentages(), team_colors)

        if is_img:

            output_path = "output/result.jpg"

            cv2.imwrite(output_path, img)

            print(f"Saved -> {output_path}")

            break

        if a.save:

            if writer is None:

                output_path = "output/result.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps / a.vid_stride,
                    (img.shape[1], img.shape[0])
                )

                if not writer.isOpened():

                    raise RuntimeError("Não foi possível criar o VideoWriter.")

            writer.write(img)

        frames += 1

        if a.stream:

            cv2.imshow(
                "YOLO Detection",
                img
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        elif frames % 30 == 0:

            print(
                f"frames: {frames}",
                end="\r"
            )

if writer:

    writer.release()

    print(
        f"\nVideo saved -> output/result.mp4 "
        f"({frames} frames)"
    )

cv2.destroyAllWindows()

if device.startswith("cuda"):
    torch.cuda.empty_cache()
